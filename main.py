import asyncio
import math

import cv2 as cv
from sklearn.neighbors import KDTree


class MyNode:
    def __init__(self, points, idx, center):
        self.points = points
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
    thickness = 2
    cv.line(image, start_point, end_point, color, thickness)


def create_graph(contours, hierarchy, precision=0.001):
    all_nodes = {}
    without_parent = []
    for idx, c in enumerate(contours):
        epsilon = precision * cv.arcLength(c, True)
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
    return without_parent, all_nodes


async def make_cuts(node: MyNode, image, tree, all_points, point_nodes, node_points):
    already_has = set()
    all_children = node.children.values()
    all_curr_nodes = [node]
    for n in all_children:
        all_curr_nodes.append(n)
    await asyncio.gather(*[make_cuts(n, image, tree, all_points, point_nodes, node_points) for n in all_children])
    if len(all_children) > 0:
        for curr_node in all_curr_nodes:
            # find the closest points
            query_result = tree.query(node_points[curr_node], 10, return_distance=False)
            min_length_for_parent = float('inf')
            min_length_point_a = []
            min_length_point_b = []
            connect_to_id = 0
            # take all curr node points and finding the smallest distance
            for i in range(0, len(curr_node.points)):
                pnt = curr_node.points[i]
                for point_id in query_result[i]:
                    need_point = all_points[point_id]
                    need_node = point_nodes[need_point]
                    if need_node.idx != curr_node.idx and (need_node.idx, curr_node.idx) not in already_has:
                        curr_length = get_length(pnt[0], all_points[point_id])
                        if min_length_for_parent > curr_length:
                            min_length_for_parent = curr_length
                            min_length_point_a = pnt[0]
                            min_length_point_b = all_points[point_id]
                            connect_to_id = need_node.idx
                            # if we found something - that's end for this point
                            break
            if len(min_length_point_b) > 0:
                add_cuttings(image, min_length_point_a, min_length_point_b)
                already_has.add((curr_node.idx, connect_to_id))


async def main():
    # choose image here
    image = cv.imread("test_files/input/fine.png")
    contours, hierarchy = read_points(image)
    # prepare some data for main algorithm
    root, all_nodes = create_graph(contours, hierarchy, precision=0.001)
    all_points = []
    point_node = {}
    node_points = {}
    for node in all_nodes.values():
        node_points[node] = []
        for point in node.points:
            p = (point[0][0], point[0][1])
            all_points.append(p)
            point_node[p] = node
            node_points[node].append(p)
    # creating tree
    tree = KDTree(all_points)
    # call main algorithm
    # this check may be reduced but cv read some useless contours
    if len(root) == 0:
        await asyncio.gather(
            *[make_cuts(n, image, tree, all_points, point_node, node_points) for n in root[0].children.values()])
    else:
        await asyncio.gather(*[make_cuts(n, image, tree, all_points, point_node, node_points) for n in root])

    # choose output here
    cv.imwrite('test_files/output/res2.jpeg', image)


if __name__ == '__main__':
    asyncio.run(main())
