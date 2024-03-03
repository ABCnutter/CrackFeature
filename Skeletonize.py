import os
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import find_contours, regionprops, label
from skimage.morphology import skeletonize, closing, square
from sklearn.neighbors import KDTree


def load_image_list(image_dir):
    if not os.path.exists(image_dir):
        raise FileNotFoundError("image_dir not found!")

    image_path_list = []
    image_name_list = []

    for img in os.listdir(image_dir):
        image_path_list.append(os.path.join(image_dir, img))
        image_name_list.append(img)

    return image_path_list, image_name_list


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


def calcu_dis_from_ctrlpts(ctrlpts):
    if ctrlpts.shape[1] == 4:
        return np.sqrt(np.sum((ctrlpts[:, 0:2] - ctrlpts[:, 2:4]) ** 2, axis=1))
    else:
        return np.sqrt(np.sum((ctrlpts[:, [0, 2]] - ctrlpts[:, [3, 5]]) ** 2, axis=1))


def get_crack_ctrlpts(centers, normals, bpoints, hband=5, vband=2, est_width=0):
    # main algorithm to obtain crack width
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)

    xmatrix = np.array([[0, 1], [-1, 0]])
    cnormalsx = np.dot(xmatrix, cnormals.T).T  # the normal of x axis
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx, ny])
            bpoints_loc = np.dot(tform, bpoints.T).T
            cpoints_loc = np.dot(tform, cpoints.T).T
            ci = cpoints_loc[i]

            bl_ind = (bpoints_loc[:, 0] - (ci[0] - hband)) * (
                bpoints_loc[:, 0] - ci[0]
            ) < 0
            br_ind = (bpoints_loc[:, 0] - ci[0]) * (
                bpoints_loc[:, 0] - (ci[0] + hband)
            ) <= 0
            bl = bpoints_loc[bl_ind]  # left points
            br = bpoints_loc[br_ind]  # right points

            if est_width > 0:
                # 下面的数值 est_width 是预估计的裂缝宽度
                half_est_width = est_width / 2
                blt = bl[(bl[:, 1] - (ci[1] + half_est_width)) * (bl[:, 1] - ci[1]) < 0]
                blb = bl[(bl[:, 1] - (ci[1] - half_est_width)) * (bl[:, 1] - ci[1]) < 0]
                brt = br[(br[:, 1] - (ci[1] + half_est_width)) * (br[:, 1] - ci[1]) < 0]
                brb = br[(br[:, 1] - (ci[1] - half_est_width)) * (br[:, 1] - ci[1]) < 0]
            else:
                blt = bl[bl[:, 1] > np.mean(bl[:, 1])]
                if np.ptp(blt[:, 1]) > vband:
                    blt = blt[blt[:, 1] > np.mean(blt[:, 1])]

                blb = bl[bl[:, 1] < np.mean(bl[:, 1])]
                if np.ptp(blb[:, 1]) > vband:
                    blb = blb[blb[:, 1] < np.mean(blb[:, 1])]

                brt = br[br[:, 1] > np.mean(br[:, 1])]
                if np.ptp(brt[:, 1]) > vband:
                    brt = brt[brt[:, 1] > np.mean(brt[:, 1])]

                brb = br[br[:, 1] < np.mean(br[:, 1])]
                if np.ptp(brb[:, 1]) > vband:
                    brb = brb[brb[:, 1] < np.mean(brb[:, 1])]

            t1 = blt[np.argsort(blt[:, 0])[-1]]
            t2 = brt[np.argsort(brt[:, 0])[0]]

            b1 = blb[np.argsort(blb[:, 0])[-1]]
            b2 = brb[np.argsort(brb[:, 0])[0]]

            interp1 = (ci[0] - t1[0]) * ((t2[1] - t1[1]) / (t2[0] - t1[0])) + t1[1]
            interp2 = (ci[0] - b1[0]) * ((b2[1] - b1[1]) / (b2[0] - b1[0])) + b1[1]

            if interp1 - ci[1] > 0 and interp2 - ci[1] < 0:
                widths.append([i, interp1 - ci[1], interp2 - ci[1]])

                interps = np.array([[ci[0], interp1], [ci[0], interp2]])

                interps_rec = np.dot(np.linalg.inv(tform), interps.T).T

                interps_rec = interps_rec.reshape(1, -1)[0, :]
                interp_segm.append(interps_rec)
        except:
            print("the %d-th was wrong" % i)
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)

    return interp_segm, widths


def estimate_normal_for_pos(pos, points, n):
    # estimate normal vectors at a given point
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(
        pos, k=n, return_distance=False, dualtree=False, breadth_first=False
    )
    normals = []
    for i in range(0, pos.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def estimate_normals(points, n):
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(
        pts, k=n, return_distance=False, dualtree=False, breadth_first=False
    )
    normals = []
    for i in range(0, pts.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def close_skeleton_contour(image_path, square_value, contours_value):
    image = io.imread(image_path)

    if len(image.shape) != 2:
        raise ValueError(
            f"Please check input images, guaranteed to be a single-channel image!"
        )
    image[image != 0] = 1
    # 使用闭运算填充小孔和连接小块
    closed_image = closing(image, square(square_value))

    skeleton_zha = skeletonize(closed_image)
    # skeleton_lee = skeletonize(closed_image, method="lee")

    # 提取轮廓
    contours = find_contours(closed_image, contours_value)
    # 创建一个空白图像
    contour_image = np.zeros_like(image)

    # 在空白图像上绘制轮廓
    for contour in contours:
        contour_image[
            np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)
        ] = 1

    skeleton_zha_contour_image = np.zeros(
        (3, image.shape[0], image.shape[1]), dtype=np.uint8
    )
    skeleton_zha_contour_image[0, skeleton_zha] = 255
    for contour in contours:
        skeleton_zha_contour_image[
            1, np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)
        ] = 255
    # skeleton_zha_contour_image[1, contour_image] = 255
    skeleton_zha_contour_image = np.transpose(skeleton_zha_contour_image, (1, 2, 0))

    # skeleton_lee_contour_image = np.zeros(
    #     (3, image.shape[0], image.shape[1]), dtype=np.uint8
    # )
    # skeleton_lee_contour_image[0, skeleton_lee] = 255
    # for contour in contours:
    #     skeleton_lee_contour_image[
    #         1, np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)
    #     ] = 255
    # # skeleton_lee_contour_image[1, contour_image] = 255
    # skeleton_lee_contour_image = np.transpose(skeleton_lee_contour_image, (1, 2, 0))

    return (
        image,
        closed_image,
        skeleton_zha,
        # skeleton_lee,
        contour_image,
        skeleton_zha_contour_image,
        # skeleton_lee_contour_image,
    )


def calculate_crack_length(skeletons, scale_factor):

    # 标记连通区域
    labeled_skeletons = label(skeletons)

    # 存储所有裂隙的像素坐标、微分裂隙的长度和像素个数
    all_crack_coordinates = []
    all_crack_lengths = []
    all_crack_pixel_counts = []

    # 处理每个连通域
    for skeleton_idx in range(
        1, np.max(labeled_skeletons) + 1
    ):  # 从1开始，因为0表示背景
        # 提取当前连通域的骨架
        current_skeleton = (labeled_skeletons == skeleton_idx).astype(np.uint8)

        # # 提取当前骨架的坐标和像素个数
        coords = np.column_stack(np.where(current_skeleton))
        pixel_count = np.sum(current_skeleton)

        # 提取当前骨架的轮廓
        contour = find_contours(current_skeleton, level=0.5)[0]

        # 裂隙的像素坐标
        crack_coordinates = np.round(contour).astype(int)

        # # 微分法求取单条裂隙的长度
        # dx = np.diff(crack_coordinates[:, 1])
        # dy = np.diff(crack_coordinates[:, 0])
        # distances = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(current_skeleton) * scale_factor
        # 将每一微分段裂隙单元的长度进行累加，得到单条裂隙的总长度
        # total_length = np.sum(distances)

        # 根据比例缩放
        # total_length *= scale_factor
        print(f"crack connected domain-{skeleton_idx} average lengths:{total_length}")

        # 存储当前裂隙的信息
        all_crack_coordinates.append(crack_coordinates)
        all_crack_pixel_counts.append(pixel_count)
        all_crack_lengths.append([skeleton_idx, total_length])

    all_crack_average_lengths = np.sum(all_crack_lengths) / len(all_crack_lengths)
    print(f"crack average lengths:{all_crack_average_lengths}")

    return (
        all_crack_coordinates,
        all_crack_lengths,
        all_crack_pixel_counts,
        all_crack_average_lengths,
    )

    # # 将每一微分段骨架单元的长度进行累加，得到骨架的总长度
    # total_length = np.sum(skeleton) * scale_factor
    # # total_length = np.sum(distances)

    # return total_length


def calculate_crack_width(
    image,
    skeletons,
    scale_factor,
    tree_n=3,
    find_contours_level=0.5,
    hband=5,
    vband=5,
    est_width=50,
):

    # 标记连通区域
    labeled_skeletons = label(skeletons)

    # 存储所有裂隙的像素坐标、微分裂隙的长度和像素个数
    all_crack_interps = []
    all_crack_widths = []
    all_connected_domain_widths = []
    # 处理每个连通域
    for skeleton_idx in range(
        1, np.max(labeled_skeletons) + 1
    ):  # 从1开始，因为0表示背景
        # 提取当前连通域的骨架
        current_skeleton = (labeled_skeletons == skeleton_idx).astype(np.uint8)

        # 提取当前骨架的坐标和像素个数
        coords = np.column_stack(np.where(current_skeleton))
        pixel_count = np.sum(current_skeleton)
        normals = estimate_normals(coords, tree_n)

        # 提取当前连通域的轮廓
        labeled_images = label(image)
        if np.max(labeled_skeletons) != np.max(labeled_images):
            raise ValueError(f"the number of images != the number of skeletons")

        current_connected_domain_coords = np.column_stack(
            np.where((labeled_images == skeleton_idx).astype(np.uint8))
        )
        current_connected_domain = np.zeros_like(labeled_images)
        current_connected_domain[
            current_connected_domain_coords[:, 0], current_connected_domain_coords[:, 1]
        ] = 1
        contours = find_contours(current_connected_domain, level=find_contours_level)

        bpoints = np.vstack(contours)

        interps, widths = get_crack_ctrlpts(
            coords, normals, bpoints, hband, vband, est_width
        )
        # print(widths)
        # if len(widths) == 0 or len(interps) == 0:

        per_crack_widths = np.sum(np.abs(widths[:, 1:]), axis=1)
        # 计算当前裂缝连通域的宽度平均值
        average_crack_widths = np.sum(per_crack_widths) / len(widths) * scale_factor
        print(
            f"crack connected domain-{skeleton_idx} average widths:{average_crack_widths}"
        )

        all_crack_interps.append(interps)
        all_crack_widths.append(widths)
        all_connected_domain_widths.append([skeleton_idx, average_crack_widths])

    # 计算图像中的所有裂缝的宽度平均值
    all_average_crack_widths = np.sum(
        np.array(all_connected_domain_widths)[:, 1]
    ) / len(all_connected_domain_widths)
    print(f"crack average widths:{all_average_crack_widths}")

    return (
        all_crack_interps,
        all_crack_widths,
        all_connected_domain_widths,
        all_average_crack_widths,
    )


def draw_widths(image, interps, show_points_nums, save_dir, image_name, plugin):
    # inters 为 all_crack_interps
    fig, ax = plt.subplots()

    for interp in interps:
        show_points_nums_copy = np.copy(show_points_nums)
        if interp.shape[0] < show_points_nums_copy:
            show_points_nums_copy = interp.shape[0]
        interps_show = interp[
            np.random.choice(interp.shape[0], show_points_nums_copy, replace=False), :
        ]

        for i in range(interps_show.shape[0]):
            ax.plot(
                [interps_show[i, 1], interps_show[i, 3]],
                [interps_show[i, 0], interps_show[i, 2]],
                c="c",
                ls="-",
                lw=2,
                marker="o",
                ms=4,
                mec="c",
                mfc="c",
            )
    ax.imshow(image)
    # 设置坐标轴不可见
    ax.axis("off")
    # 调整子图的间距，使得图像紧凑显示
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    save_file_dir = os.path.join(save_dir, image_name.split(".")[0])
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    plt.savefig(
        os.path.join(save_file_dir, image_name.split(".")[0] + f"{plugin}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # 显示结果
    # plt.show()


def radians_to_degrees_minutes_seconds(angle_radians):
    # 转换弧度为度
    angle_degrees = math.degrees(angle_radians)

    # 提取度、分、秒
    degrees = int(angle_degrees)
    remainder_minutes = (angle_degrees - degrees) * 60
    minutes = int(remainder_minutes)
    seconds = (remainder_minutes - minutes) * 60

    return degrees, minutes, seconds


def estimate_crack_direction(image):
    # 标记裂缝连通域
    labeled_image = label(image)

    # 存储裂缝的法线方向和椭圆信息
    crack_connected_domain_directions = []

    # 处理每个连通域
    for region in regionprops(labeled_image):

        current_crack_direction = region.orientation

        degrees, minutes, seconds = radians_to_degrees_minutes_seconds(
            current_crack_direction
        )

        print(
            f"{region.label}-{current_crack_direction}-子连通域弧度对应的度分秒为：{degrees}° {minutes}' {seconds}\""
        )

        crack_connected_domain_directions.append(
            [region.label, current_crack_direction, degrees, minutes, seconds]
        )

        # 计算图像中的所有裂缝的面积平均值
    all_average_crack_directions = np.sum(
        np.array(crack_connected_domain_directions)[:, 1]
    ) / len(crack_connected_domain_directions)

    all_average_degrees, all_average_minutes, all_average_seconds = (
        radians_to_degrees_minutes_seconds(all_average_crack_directions)
    )
    print(
        f"整体平均弧度-{all_average_crack_directions}-对应的度分秒为：{all_average_degrees}° {all_average_minutes}' {all_average_seconds}\""
    )
    all_average_crack_direction = [
        all_average_crack_directions,
        all_average_degrees,
        all_average_minutes,
        all_average_seconds,
    ]
    return crack_connected_domain_directions, all_average_crack_direction


def extract_crack_areas(image, scale_factor=1):
    # 标记裂缝连通域
    labeled_image = label(image)

    # 存储裂缝的法线方向和椭圆信息
    crack_connected_domain_areas = []

    # 处理每个连通域
    for region in regionprops(labeled_image):

        crack_area = region.area * (scale_factor * scale_factor)
        print(f'{region.label}-子连通域面积为：{crack_area}"')
        crack_connected_domain_areas.append([region.label, crack_area])

    # 计算图像中的所有裂缝的面积平均值
    all_average_crack_areas = np.sum(
        np.array(crack_connected_domain_areas)[:, 1]
    ) / len(crack_connected_domain_areas)
    print(f"crack average areas:{all_average_crack_areas}")

    return crack_connected_domain_areas, all_average_crack_areas


def save_results(save_dir, image_name, image, plugin):
    save_file_dir = os.path.join(save_dir, image_name.split(".")[0])
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    io.imsave(
        fname=os.path.join(save_file_dir, image_name.split(".")[0] + f"{plugin}.png"),
        arr=image,
    )


def extract_crack_feature_parameters(
    image_dir,
    save_dir,
    square_value,
    contours_value,
    scale_factor,
    tree_n=3,
    find_contours_level=0.5,
    hband=5,
    vband=5,
    est_width=50,
    show_points_nums=50,
):
    img_path_list, img_name_list = load_image_list(image_dir)
    for img_path, img_name in tqdm(zip(img_path_list, img_name_list)):
        try:
            print(
                f"***************************************************************************************************"
            )
            print(
                f"**************************************** {img_name} BEGIN! ****************************************"
            )
            print(
                f"***************************************************************************************************"
            )

            ########################################## calculate_crack_closed_skeleton ##########################################

            (
                image,
                closed_image,
                skeleton_zha,
                # skeleton_lee,
                contour_image,
                skeleton_zha_contour_image,
                # skeleton_lee_contour_image,
            ) = close_skeleton_contour(img_path, square_value, contours_value)
            save_results(save_dir, img_name, image * 255, "_image")
            save_results(save_dir, img_name, closed_image * 255, "_closed_image")
            save_results(save_dir, img_name, skeleton_zha, "_skeleton_zha")
            # save_results(save_dir, img_name, skeleton_lee * 255, "_skeleton_lee")
            save_results(save_dir, img_name, contour_image * 255, "_contour_image")
            save_results(
                save_dir,
                img_name,
                skeleton_zha_contour_image,
                "_skeleton_zha_contour_image",
            )
            # save_results(save_dir, img_name, skeleton_lee_contour_image, '_skeleton_lee_contour_image')

            save_file_dir = os.path.join(save_dir, img_name.split(".")[0])
            if not os.path.exists(save_file_dir):
                os.makedirs(save_file_dir)

            ########################################## calculate_crack_length ##########################################

            (
                all_crack_coordinates,
                all_crack_lengths,
                all_crack_pixel_counts,
                all_crack_average_lengths,
            ) = calculate_crack_length(skeleton_zha, scale_factor)
            # 将lengths逐行写入文本文件
            with open(f"{save_file_dir}/{img_name}_crack_length.txt", "w") as file:
                for row in all_crack_lengths:
                    file.write(
                        "_crack_connected_domain_average_length: ".join(map(str, row))
                        + "\n"
                    )

                file.write(
                    f"total figure crack average length: {all_crack_average_lengths}"
                )

            ########################################## calculate_crack_width ##########################################

            print(f"{img_name} BEGIN!")
            (
                all_crack_interps,
                all_crack_widths,
                all_connected_domain_widths,
                all_average_crack_widths,
            ) = calculate_crack_width(
                closed_image,
                skeleton_zha,
                scale_factor,
                tree_n,
                find_contours_level,
                hband,
                vband,
                est_width,
            )
            # 将widths逐行写入文本文件
            with open(f"{save_file_dir}/{img_name}_crack_width.txt", "w") as file:
                for row in all_connected_domain_widths:
                    file.write(
                        "_crack_connected_domain_average_width: ".join(map(str, row))
                        + "\n"
                    )

                file.write(
                    f"total figure crack average width: {all_average_crack_widths}"
                )

            draw_widths(
                skeleton_zha_contour_image,
                all_crack_interps,
                show_points_nums,
                save_dir,
                img_name,
                "_width",
            )

            ########################################## calculate_crack_direction ##########################################

            crack_connected_domain_directions, all_average_crack_direction = (
                estimate_crack_direction(closed_image)
            )
            # 将direction逐行写入文本文件
            with open(f"{save_file_dir}/{img_name}_crack_direction.txt", "w") as file:
                for row in crack_connected_domain_directions:
                    file.write(
                        f"{row[0]}_crack_connected_domain_average_direction:{row[1]} --- 对应的度分秒为：{row[2]}° {row[3]}' {row[4]}\""
                        + "\n"
                    )

                file.write(
                    f"total figure crack average direction: {all_average_crack_direction[0]} --- 对应的度分秒为：{all_average_crack_direction[1]}° {all_average_crack_direction[2]}' {all_average_crack_direction[3]}"
                )

            ########################################## calculate_crack_area ##########################################

            crack_connected_domain_areas, all_average_crack_areas = extract_crack_areas(
                closed_image, scale_factor
            )
            # 将area逐行写入文本文件
            with open(f"{save_file_dir}/{img_name}_crack_area.txt", "w") as file:
                for row in crack_connected_domain_areas:
                    file.write(
                        "_crack_connected_domain_average_area: ".join(map(str, row))
                        + "\n"
                    )

                file.write(
                    f"total figure crack average area: {all_average_crack_areas}"
                )

            print(
                f"####################################################################################################"
            )
            print(
                f"######################################### {img_name} OVER! #########################################"
            )
            print(
                f"####################################################################################################"
            )
        except:
            import logging

            logging.error(
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  {img_name} exists wrong  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            continue


if __name__ == "__main__":

    import configparser

    # 将参数保存到配置文件
    config = configparser.ConfigParser()
    config["Parameters"] = {
        "square_value": "20",
        "contours_value": "0.8",
        "scale_factor": "2.6",
        "tree_n": "3",
        "find_contours_level": "0.5",
        "hband": "5",
        "vband": "5",
        "est_width": "30",
        "show_points_nums": "150",
        "image_dir": "test",  # 存放所有裂缝二值图片的文件夹
        "save_dir": "results",
    }
    # 保存配置文件
    # 保存配置文件到 save_dir 路径下
    save_file_dir = config["Parameters"]["save_dir"]
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    config_file_path = os.path.join(config["Parameters"]["save_dir"], "config.ini")
    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    # 读取配置文件中的参数
    config = configparser.ConfigParser()
    config.read(config_file_path)

    square_value = int(config["Parameters"]["square_value"])
    contours_value = float(config["Parameters"]["contours_value"])
    scale_factor = float(config["Parameters"]["scale_factor"])
    tree_n = int(config["Parameters"]["tree_n"])
    find_contours_level = float(config["Parameters"]["find_contours_level"])
    hband = int(config["Parameters"]["hband"])
    vband = int(config["Parameters"]["vband"])
    est_width = int(config["Parameters"]["est_width"])
    show_points_nums = int(config["Parameters"]["show_points_nums"])
    image_dir = config["Parameters"]["image_dir"]
    save_dir = config["Parameters"]["save_dir"]

    # 然后在你的函数中使用这些参数
    extract_crack_feature_parameters(
        image_dir,
        save_dir,
        square_value,
        contours_value,
        scale_factor,
        tree_n,
        find_contours_level,
        hband,
        vband,
        est_width,
        show_points_nums,
    )
