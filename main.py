from typing import Union, Optional
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


class Silo3D:
    """
    基于激光雷达的筒仓物料体积计算，选择Delaunay三角形模型能够覆盖各种不规则的表面
    1 如果 仓体内已经有物料， 则用户可以提供筒仓高、半径方式, 但是推荐 建立仓体模型 来替代 空仓点云
    2 如果空仓 使用激光雷达扫描空仓点云，精度更高
    3 点云格式 x y z intensity r g b
    """

    def __init__(self, silo_height=30,
                 silo_radius=5,
                 grid_size=0.05,
                 n_neighbors=20,
                 radius=0.1,
                 alpha=1.5,
                 n=5,
                 empty_pcd: str = None):
        """
        :param silo_height: 假设仓体是圆柱形 仓体高度，
        :param silo_radius: 假设仓体是圆柱形  仓体半径
        :param grid_size: 网格大小
        :param n_neighbors: 统计滤波参数 邻居个数
        :param radius: 半径滤波参数 半径
        :param alpha: 统计滤波参数 距离超过 alpha * 平均距离为异常
        :param n:半径滤波参数， 半径内个数小于 n + 1 为异常
        :param empty_pcd: 仓体空时激光雷达的扫描后的3d点云，此时silo_height和silo_radius失效
        """
        self.silo_height = silo_height
        self.silo_radius = silo_radius
        self.grid_size = grid_size
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.v_silo = np.pi * self.silo_radius ** 2 * self.silo_height
        self.alpha = alpha
        self.n = n
        self.empty_pcd = empty_pcd
        return

    def __call__(self, points: Union[str, np.ndarray]):
        if isinstance(points, str):
            points = self.load_cloud(points)

        points = self.remove_noise(points)

        if self.empty_pcd is not None:
            empty_pts = self.remove_noise(self.load_cloud(self.empty_pcd))
            v_material = self.calculate_volume_by_empty(points, empty_pts)
        else:
            v_material = self.calculate_volume(points)

        return v_material

    def calculate_volume_by_empty(self, current_pts: np.ndarray, empty_pts: Optional[np.ndarray] = None):
        empty_surface = self.build_surface(empty_pts)
        material_surface = self.build_surface(current_pts)

        # 公告区域
        x_min = max(
            empty_pts[:, 0].min(),
            current_pts[:, 0].min()
        )

        x_max = min(
            empty_pts[:, 0].max(),
            current_pts[:, 0].max()
        )

        y_min = max(
            empty_pts[:, 1].min(),
            current_pts[:, 1].min()
        )

        y_max = min(
            empty_pts[:, 1].max(),
            current_pts[:, 1].max()
        )

        # 生成积分网格
        x = np.arange(x_min, x_max, self.grid_size)

        y = np.arange(y_min, y_max, self.grid_size)

        xx, yy = np.meshgrid(x, y)

        # 根据网格 插值出z, 同时如果网格网格点不再三角形内，则返回哪np.nan
        z_bottom = empty_surface(xx, yy)
        z_material = material_surface(xx, yy)

        # 高度差 空仓的表面要大于等于物料（比如才开始倒料时，物料还没覆盖整个空仓面)
        valid_mask = (~np.isnan(z_bottom) & ~np.isnan(z_material))
        height = np.zeros_like(z_bottom)
        height[valid_mask] = (z_material[valid_mask] - z_bottom[valid_mask])

        # height = (z_material - z_bottom)
        # height[np.isnan(height)] = 0
        # height[height < 0] = 0

        # 覆盖率
        coverage = (np.sum(valid_mask) / valid_mask.size)

        # 积分求体积
        cell_area = (self.grid_size * self.grid_size)
        volume = np.sum(height * cell_area)
        return volume

    def calculate_volume(self, points: np.ndarray):
        # 计算仓体中心
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        r = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        mask = r < self.silo_radius
        material_pts = points[mask]

        # TIN三角网（兼容各种表面)
        xy = material_pts[:, 0:2]
        tri = Delaunay(xy)

        # 空仓体积
        v_empty = 0.0
        for simplex in tri.simplices:
            p1 = material_pts[simplex[0]]
            p2 = material_pts[simplex[1]]
            p3 = material_pts[simplex[2]]

            x1, y1, z1 = p1
            x2, y2, z2 = p2
            x3, y3, z3 = p3

            # 向量的差积等于平行四边形的面积
            # a = (x2 - x1), (y2 - y1)
            # b = (x3 - x1), (y3 - y1)
            area = 0.5 * abs(
                (x2 - x1) * (y3 - y1)
                - (x3 - x1) * (y2 - y1)
            )

            # 求三个点的z的均值
            empty_h = ((self.silo_height - z1) + (self.silo_height - z2) + (self.silo_height - z3)) / 3.0

            # 实际是求这个三角形区域的高度积分 V = sum(dh * da) 对线性平面 V = area * mean(h)
            v_empty += area * empty_h

        return self.v_silo - v_empty

    @staticmethod
    def build_surface(points):
        xy = points[:, 0:2]
        z = points[:, 2]

        # tri = Delaunay(xy)

        # LinearNDInterpolator 根据 xy 构建了三角形, tri 感觉不需要了
        interp = LinearNDInterpolator(
            xy,
            z,
            fill_value=np.nan
        )
        return interp

    @staticmethod
    def load_cloud(filename):
        data = np.loadtxt(filename)
        return data[:, :3]

    def remove_noise(self, points: np.ndarray):
        """
        :param points:
        :return:
        """

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=self.n_neighbors,
                std_ratio=self.alpha
            )
            points = np.asarray(pcd.points)

        except Exception as err:
            points = self.statistical_outlier_removal(points)
            points = self.radius_outlier_removal(points)

        return points

    def statistical_outlier_removal(self, points: np.ndarray):
        """
        统计滤波
        :param points:
        :return:
        """
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors).fit(points)
        distances, _ = neigh.kneighbors(points)
        mean_dist = distances[:, 1:].mean(axis=1)
        mask = mean_dist < np.mean(mean_dist) + self.alpha * np.std(mean_dist)
        return points[mask]

    def radius_outlier_removal(self, points: np.ndarray):
        """
        半径滤波： 指定半径内点的数量太少判为噪声
        :param points:
        :return:
        """
        neigh = NearestNeighbors(radius=self.radius).fit(points)
        counts = neigh.radius_neighbors(points, return_distance=False)
        mask = np.array([len(c) >= self.n + 1 for c in counts])
        return points[mask]


if __name__ == '__main__':
    silo3d = Silo3D()
    silo3d('/home/thinkbook/workspace/silo3d/3d/biaomian5.asc')
