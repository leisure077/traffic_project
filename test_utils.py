# -*- coding: utf-8 -*-
"""
test_utils.py —— 课时 58 §三.4 建议优先测试的纯函数

覆盖四组：haversine（同点为0/单位换算/已知距离）、kmeans（两团能分开、中心不出 NaN、
同坐标不拆簇）、choose_best_k（全同点不拆 / 真远点仍拆）、
decide_purpose（居住/购物/旅游/其他 + 窗口覆盖 bug 回归）、毫秒时间戳 +8 转换。

跑法：
    python -m unittest test_utils.py -v
    python test_utils.py
"""
import unittest

import numpy as np
import pandas as pd

import config
import data_clean
from purpose import decide_purpose
from tr_stage2 import choose_best_k, kmeans
from utils import haversine, is_night, parse_time, project_to_meters

# 太原街 → 中街，沈阳两条商业街的真实坐标，实际约 2.1 km
TYJ = (123.4119, 41.8052)
ZJ = (123.4370, 41.8033)


class TestHaversine(unittest.TestCase):
    def test_same_point_is_zero(self):
        self.assertEqual(haversine(123.4, 41.8, 123.4, 41.8), 0.0)

    def test_known_distance(self):
        d = haversine(*TYJ, *ZJ)
        self.assertAlmostEqual(d, 2.1, delta=0.3)

    def test_unit_m_equals_km_times_1000(self):
        km = haversine(*TYJ, *ZJ)
        m = haversine(*TYJ, *ZJ, unit="m")
        self.assertAlmostEqual(m, km * 1000, places=6)

    def test_symmetry(self):
        self.assertAlmostEqual(haversine(*TYJ, *ZJ), haversine(*ZJ, *TYJ), places=9)

    def test_vectorized_matches_scalar(self):
        """广播数组的每个元素必须与同坐标的单点调用一致（课时 11 的广播口径）。"""
        lon1 = np.array([123.40, 123.45])
        lat1 = np.array([41.80, 41.82])
        lon2 = np.array([123.43, 123.50])
        lat2 = np.array([41.81, 41.79])
        vec = haversine(lon1, lat1, lon2, lat2)
        for i in range(2):
            self.assertAlmostEqual(float(vec[i]),
                                   haversine(lon1[i], lat1[i], lon2[i], lat2[i]), places=9)


class TestTime(unittest.TestCase):
    def test_parse_time(self):
        ts = parse_time(pd.Series(["20181003123000"]))
        self.assertEqual(ts.iloc[0].year, 2018)
        self.assertEqual(ts.iloc[0].hour, 12)

    def test_is_night_boundaries(self):
        """NIGHT_HOURS = [(0,6),(20,23)]，两端都是闭区间，所以 6 点和 23 点算夜间。"""
        for h in (0, 3, 6, 20, 22, 23):
            self.assertTrue(is_night(h), f"{h} 点应属夜间")
        for h in (7, 8, 12, 19):
            self.assertFalse(is_night(h), f"{h} 点不应属夜间")

    def test_ms_timestamp_shifts_by_8_hours(self):
        """13 位 UTC 毫秒戳必须 +8 小时变北京时间；去掉 +8 这条就红。"""
        beijing = pd.Timestamp("2018-10-03 00:41:22")
        utc_ms = int((beijing - pd.Timedelta(hours=config.RAW_TIMEZONE_OFFSET_HOURS))
                     .value // 10 ** 6)
        df = pd.DataFrame({"time": [str(utc_ms)]})
        out = data_clean.rule4_timestamp(df.copy())
        self.assertEqual(out["time"].iloc[0], "20181003004122")
        self.assertTrue(out["time"].iloc[0].startswith(config.TARGET_DATE))

    def test_text_timestamp_passes_through(self):
        df = pd.DataFrame({"time": ["20181003083000"]})
        out = data_clean.rule4_timestamp(df.copy())
        self.assertEqual(out["time"].iloc[0], "20181003083000")


class TestKmeans(unittest.TestCase):
    def test_two_blobs_separate(self):
        """课时 36 §五：两团明显分离的点必须分成两簇。走生产路径，先投影成米。"""
        np.random.seed(0)
        pts = np.vstack([np.random.normal([123.40, 41.80], 0.005, (50, 2)),
                         np.random.normal([123.60, 41.95], 0.005, (50, 2))])
        labels, centers = kmeans(project_to_meters(pts[:, 0], pts[:, 1]), 2)
        self.assertEqual(len(set(labels.tolist())), 2)
        same_block = labels[:50] == labels[49]
        self.assertTrue(same_block.all() or (~same_block).all())

    def test_no_nan_centers(self):
        """空簇要保留旧中心，不能对空集取均值得到 NaN（课时 36 §六 的常见 bug）。"""
        rng = np.random.default_rng(3)
        pts = np.vstack([rng.normal([0, 0], 0.01, (30, 2)),
                         rng.normal([9, 9], 0.01, (30, 2))])
        for k in (2, 3, 5, 8):
            _, centers = kmeans(pts, k)
            self.assertFalse(np.isnan(centers).any(), f"k={k} 出现 NaN 中心")

    def test_k_ge_n_each_point_own_cluster(self):
        pts = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 1.0]])
        labels, _ = kmeans(pts, 5)
        self.assertEqual(sorted(labels.tolist()), [0, 1, 2])

    def test_k_ge_n_duplicate_points_share_cluster(self):
        """簇数超过点数时，坐标完全相同的两个点必须落在同一簇。

        同一个基站的点经纬度一模一样，按点编号发簇号会把它们拆成两簇，
        阶段3 就会把「一直待在同一个基站」判成出行。
        """
        pts = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
        labels, _ = kmeans(pts, 5)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(sorted(set(labels.tolist())), [0, 1])


class TestChooseBestK(unittest.TestCase):
    def test_identical_points_stay_one_cluster(self):
        """坐标全相同的两个点，任何 K 下都不许被拆开（退化拆法的轮廓得分恰好 0，不能赢过基线）。"""
        np.random.seed(config.RANDOM_STATE)
        pts = np.array([[123.4, 41.8], [123.4, 41.8]])
        X = project_to_meters(pts[:, 0], pts[:, 1])
        labels, best_k = choose_best_k(X)
        self.assertEqual(best_k, 1)
        self.assertEqual(set(labels.tolist()), {0})

    def test_two_far_points_still_split(self):
        """基线改成 0 分之后，真正隔得远的两个点仍要拆开，别把修复做成「一律不拆」。"""
        np.random.seed(config.RANDOM_STATE)
        pts = np.array([[123.30, 41.80], [123.60, 41.80]])
        X = project_to_meters(pts[:, 0], pts[:, 1])
        labels, best_k = choose_best_k(X)
        self.assertEqual(best_k, 2)
        self.assertNotEqual(labels[0], labels[1])

    def test_reproducible_with_same_seed(self):
        """同一种子下两次聚类必须逐位一致。注意造点本身也吃随机流，所以每次聚类前都要重新播种。"""
        np.random.seed(0)
        pts = np.random.rand(60, 2) * 0.2
        np.random.seed(config.RANDOM_STATE)
        a, _ = kmeans(pts, 3)
        np.random.seed(config.RANDOM_STATE)
        b, _ = kmeans(pts, 3)
        self.assertTrue((a == b).all())


class TestDecidePurpose(unittest.TestCase):
    """课时 45：修好「购物窗覆盖旅游窗」的 bug 之后，旅游分支必须能命中。"""

    @staticmethod
    def _row(**kw):
        base = {"start_hour": 12, "end_hour": 14, "duration_hours": 2,
                "nearest_poi_type": None, "match_success": False}
        base.update(kw)
        return base

    def test_long_stay_at_night_is_living(self):
        self.assertEqual(decide_purpose(self._row(duration_hours=8,
                                                  start_hour=23, end_hour=6)), "居住")

    def test_shopping_poi_beats_time_window(self):
        self.assertEqual(decide_purpose(self._row(match_success=True,
                                                  nearest_poi_type="购物")), "购物")

    def test_travel_poi(self):
        self.assertEqual(decide_purpose(self._row(match_success=True,
                                                  nearest_poi_type="旅游")), "旅游")

    def test_narrow_window_checked_first(self):
        """9~17 同时落在旅游窗和购物窗内，必须先判旅游——宽条件在前就是死代码。"""
        self.assertEqual(decide_purpose(self._row(start_hour=9, end_hour=17)), "旅游")

    def test_wide_window_falls_to_shopping(self):
        self.assertEqual(decide_purpose(self._row(start_hour=10, end_hour=20)), "购物")

    def test_outside_all_windows_is_other(self):
        self.assertEqual(decide_purpose(self._row(start_hour=7, end_hour=23)), "其他")


if __name__ == "__main__":
    unittest.main(verbosity=2)
