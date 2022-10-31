import asyncio
import math

import cv2 as cv
import numpy as np
from rtree import index


class MyNode:
    def __init__(self, points, idx, center):
        self.points = points
        #self.sorted_points = np.sort(points, axis=0)
        self.children = {}
        self.parent = None
        self.idx = idx
        self.center = center


def get_length(a, b):
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def read_points(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 150, 255, type=cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours, hierarchy


def add_cuttings(image, point_a, point_b):
    start_point = (point_a[0], point_a[1])
    end_point = (point_b[0], point_b[1])
    color = (0, 255, 0)
    thickness = 1
    cv.line(image, start_point, end_point, color, thickness)


def print_points(points):
    for p in points:
        print(p, end=" ")
    print()


def create_graph(contours, hierarchy):
    all_nodes = {}
    without_parent = []
    for idx, c in enumerate(contours):
        epsilon = 0.0034 * cv.arcLength(c, True)
        # approximate the contour
        c = cv.approxPolyDP(c, epsilon, True)
        all_nodes[idx] = (MyNode(c, idx, (0, 0)))
    for i in range(len(contours)):
        parent_index = hierarchy[0][i][3]
        if parent_index != -1:
            all_nodes[i].parent = all_nodes[parent_index]
            all_nodes[parent_index].children[i] = all_nodes[i]
        else:
            without_parent.append(all_nodes[i])
    return without_parent



async def make_cuts(node: MyNode, image):
    idx2d = index.Index()
    id_one = 0
    point_nodes = {}
    point_ids = {}
    ids_point = {}
    all_curr_nodes = [node]
    all_children = node.children.values()
    for n in all_children:
        all_curr_nodes.append(n)
    for curr_node in all_curr_nodes:
        for i in range(0, len(curr_node.points)):
            pnt = curr_node.points[i]
            idx2d.insert(id_one, coordinates=(pnt[0][0], pnt[0][1]))
            point_nodes[id_one] = curr_node
            point_ids[id_one] = pnt
            ids_point[(pnt[0][0], pnt[0][1])] = id_one
            id_one += 1
    already_has = set()
    await asyncio.gather(*[make_cuts(n, image) for n in all_children])
    if len(all_children) > 0:
        for curr_node in all_curr_nodes:
            min_length_for_parent = float('inf')
            min_length_point_a = []
            min_length_point_b = []
            connect_to_id = 0
            for i in range(0, len(curr_node.points)):
                pnt = curr_node.points[i]
                for point_id in idx2d.nearest((pnt[0][0], pnt[0][1]), 13):
                    need_node = point_nodes[point_id]
                    if need_node.idx != curr_node.idx and (need_node.idx, curr_node.idx) not in already_has:
                        curr_length = get_length(pnt[0], point_ids[point_id][0])
                        if min_length_for_parent > curr_length:
                            min_length_for_parent = curr_length
                            min_length_point_a = pnt[0]
                            min_length_point_b = point_ids[point_id][0]
                            connect_to_id = need_node.idx
            if len(min_length_point_b) > 0:
                add_cuttings(image, min_length_point_a, min_length_point_b)
                already_has.add((curr_node.idx, connect_to_id))


async def main():
    image = cv.imread("fine.png")
    contours, hierarchy = read_points(image)
    root = create_graph(contours, hierarchy)
    await asyncio.gather(*[make_cuts(n, image) for n in root[0].children.values()])
    # for n in root[0].children.values():
    #     await make_cuts(n, image)
        # for child in root[0].children.values():
    #     make_cuts(child, image)
    cv.imwrite('res.jpeg', image)
    # for c in range(1, len(contours)):
    #     for p in range(0, len(contours[c]), max(len(contours[c]) // 13, 1)):
    #         idx2d.insert(id_one, coordinates=(contours[c][p][0][0], contours[c][p][0][1]))
    #         id_one += 1
    #
    # for c in range(1, len(contours)):
    #     for p in range(0, len(contours[c]), max(len(contours[c]) // 13, 1)):
    #         for j in idx2d.nearest(coordinates=(contours[c][p][0][0], contours[c][p][0][1]), num_results=3):
    #             pass


if __name__ == '__main__':
    asyncio.run(main())
