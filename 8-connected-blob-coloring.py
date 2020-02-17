from PIL import *
from PIL import Image, ImageDraw
import numpy as np
import cv2


def main():
    img = Image.open('nums.png')
    img = img.resize((500, 500))
    img_gray = img.convert('L')  # converts the image to grayscale image
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)  # from np array to PIL format
    # im.show()

    # a_bin = binary_image(100,100, ONE)   #creates a binary image
    a_bin = np.asarray(im)
    label = blob_coloring_8_connected(a_bin, ONE)
    new_img = Image.fromarray(np.uint8(a_bin))
    new_img.show()
    new_img2 = np2PIL_color(label)
    colored_img = new_img2.copy()
    new_img2.show()
    rects = rectangles(label)
    rects_img = ImageDraw.Draw(new_img2)
    for i in range(len(rects)):
        rects_img.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
        #im1 = new_img2.crop((rects[i][1]+1, rects[i][0]+1, rects[i][3], rects[i][2]))

    colored_img.show()
    new_img2.show()
    im1 = colored_img.crop((23, 26, 56+1, 74+1))
    im1 = im1.convert('L')
    im1.show()
    # a = np.asarray(im1)
    # im2 = threshold(a, 100, ONE, 0)
    # im1 = Image.fromarray(im2)
    # im1.show()
    print(cv2.moments(np.asarray(im1)))
    moments = momentsof(np.asarray(im1))


def binary_image(nrow, ncol, Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow, ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10

    for i in range(50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i - 20][90 - i + 1] = 1
        mask_lines[i - 20][90 - i + 2] = 1
        mask_lines[i - 20][90 - i + 3] = 1

    # mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
    # mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    # mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    # mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    # imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    # imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img


def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape=im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    print("nrow, ncol", nrow, ncol)
    im = np.zeros(shape=(nrow, ncol), dtype=int)
    a = np.zeros(shape=max_label, dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            c = bim[i][j]
            l = bim[i][j - 1]
            u = bim[i - 1][j]
            label_u = im[i - 1][j]
            label_l = im[i][j - 1]
            label_ul = im[i - 1][j - 1]
            label_ur = im[i - 1][j + 1]

            im[i][j] = max_label
            if c == ONE:
                min_label = min(label_u, label_l, label_ul, label_ur)
                if min_label == max_label:
                    k += 1
                    im[i][j] = k
                else:
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label:
                        update_array(a, min_label, label_u)

                    if min_label != label_l and label_l != max_label:
                        update_array(a, min_label, label_l)

                    if min_label != label_ul and label_ul != max_label:
                        update_array(a, min_label, label_ul)

                    if min_label != label_ur and label_ur != max_label:
                        update_array(a, min_label, label_ur)

            else:
                im[i][j] = max_label
    # final reduction in label array
    for i in range(k + 1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    # second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return color_im


def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:  # a[index] == lab_small
            break

    return


def rectangles(labelsarr):
    nrow = labelsarr.shape[0]
    ncol = labelsarr.shape[1]

    #xmin = 10000
    #ymin = 10000
    #xmax = 0
    #ymax = 0
    #currentlabel = 0

    # Map each different (R,G,B) tuple to a index
    k = 0
    colorindices = {}
    for i in range(nrow):
        for j in range(ncol):
            rgb = (labelsarr[i][j][0], labelsarr[i][j][1], labelsarr[i][j][2])
            if not rgb in colorindices:
                colorindices[rgb] = k
                k = k + 1

    rectangles = np.zeros(shape=(len(colorindices), 4), dtype=int)

    for x in range(len(rectangles)):
        rectangles[x][0] = 10000
        rectangles[x][1] = 10000
        rectangles[x][2] = 0
        rectangles[x][3] = 0

    for i in range(nrow):
        for j in range(ncol):
            rgb = (labelsarr[i][j][0], labelsarr[i][j][1], labelsarr[i][j][2])
            if i < rectangles[colorindices.get(rgb)][0]:
                rectangles[colorindices.get(rgb)][0] = i
            if j < rectangles[colorindices.get(rgb)][1]:
                rectangles[colorindices.get(rgb)][1] = j
            if i > rectangles[colorindices.get(rgb)][2]:
                rectangles[colorindices.get(rgb)][2] = i
            if j > rectangles[colorindices.get(rgb)][3]:
                rectangles[colorindices.get(rgb)][3] = j

    print(rectangles)

    return rectangles

def momentsof(image):
    rows = image.shape[0]
    cols = image.shape[1]
    raw_moments = [[0,0,0,0], [0,0,0], [0,0], [0]]

    k = 4
    for i in range(k):
        for j in range(k):
            for x in range(rows):
                for y in range(cols):
                    raw_moments[i][j] += pow(x, i) * pow(y, j) * image[x][y]
        k = k-1


    central_moments = [[0,0,0,0], [0,0,0], [0,0], [0]]
    xbar = raw_moments[1][0] / raw_moments[0][0]
    ybar = raw_moments[0][1] / raw_moments[0][0]

    # central_moments[0][0] = raw_moments[0][0]
    # central_moments[0][1] = 0
    # central_moments[1][0] = 0
    # central_moments[1][1] = raw_moments[1][1] - xbar * raw_moments[0][1]
    # central_moments[2][0] = raw_moments[2][0] - xbar * raw_moments[1][0]
    # central_moments[0][2] = raw_moments[0][2] - ybar * raw_moments[0][1]
    # central_moments[2][1] = raw_moments[2][1] - 2 * xbar * raw_moments[1][1] - ybar * raw_moments[2][0] + 2 * xbar * xbar * raw_moments[0][1]
    # central_moments[1][2] = raw_moments[1][2] - 2 * ybar * raw_moments[1][1] - xbar * raw_moments[0][2] + 2 * ybar * ybar * raw_moments[1][0]
    # central_moments[3][0] = raw_moments[3][0] - 3 * xbar * xbar * xbar * raw_moments[2][0] + 2 * xbar * xbar * raw_moments[1][0]
    # central_moments[0][3] = raw_moments[0][3] - 3 * ybar * ybar * ybar * raw_moments[0][2] + 2 * ybar * ybar * raw_moments[0][1]

    k = 4
    for i in range(k):
        for j in range(k):
            for x in range(rows):
                for y in range(cols):
                    central_moments[i][j] += pow(x - xbar, i) * pow(y - ybar, j) * image[x][y]
        k = k-1


    scale_invariants = [[0,0,0,0], [0,0,0], [0,0], [0]]

    k = 4
    for i in range(k):
        for j in range(k):
            scale_invariants[i][j] = central_moments[i][j] / pow(central_moments[0][0], 1 + (i+j)/2)
        k = k - 1


    rotation_invariants = [0,0,0,0,0,0,0]

    # indices indicate I sub index+1
    rotation_invariants[0] = scale_invariants[2][0] + scale_invariants[0][2]
    rotation_invariants[1] = pow(rotation_invariants[0], 2) + 4 * pow(3*scale_invariants[1][1], 2)
    rotation_invariants[2] = pow(scale_invariants[3][0] - 3 * scale_invariants[1][2], 2) + pow(3 * scale_invariants[2][1] - scale_invariants[0][3], 2)
    rotation_invariants[3] = pow(scale_invariants[3][0] + scale_invariants[1][2], 2) + pow(scale_invariants[2][1] + scale_invariants[0][3], 2)
    rotation_invariants[4] = (scale_invariants[3][0] - 3 * scale_invariants[1][2]) * (scale_invariants[3][0] + scale_invariants[1][2]) * (pow(scale_invariants[3][0] + scale_invariants[1][2], 2) + 3 * pow(scale_invariants[2][1] + scale_invariants[0][3], 2)) + (3 * scale_invariants[2][1] - scale_invariants[0][3]) * (scale_invariants[2][1] + scale_invariants[0][3]) * (3 * pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - pow(scale_invariants[2][1] + scale_invariants[0][3], 2))
    rotation_invariants[5] = (scale_invariants[2][0] - scale_invariants[0][2]) * (pow(scale_invariants[3][0] + scale_invariants[1][2], 2) + pow(scale_invariants[2][1] + scale_invariants[0][3], 2)) + 4 * scale_invariants[1][1] * (scale_invariants[3][0] + scale_invariants[1][2]) * (scale_invariants[2][1] + scale_invariants[0][3])
    rotation_invariants[6] = (3 * scale_invariants[2][1] - scale_invariants[3][0]) * (scale_invariants[3][0] + scale_invariants[1][2]) * (pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - 3 * pow(scale_invariants[2][1] + scale_invariants[0][3], 2)) + (scale_invariants[3][0] - 3 * scale_invariants[1][2]) * (scale_invariants[2][1] + scale_invariants[0][3]) * (3 * pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - pow(scale_invariants[2][1] + scale_invariants[0][3], 2))


    return rotation_invariants


if __name__ == '__main__':
    main()
