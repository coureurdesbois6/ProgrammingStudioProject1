from PIL import Image, ImageDraw
import numpy as np

DICTIONARY_SIZE = 10 #36 if alphabet is included
THRESHOLD = 150

class ImageReader:
    def launch(self, dest):
        img = Image.open(dest)
        im = self.to_greyscale(img) #greyscale image
        a_bin = np.asarray(im)
        label = self.blob_coloring_8_connected(a_bin)
        new_img = Image.fromarray(np.uint8(a_bin))
        rects = self.rectangles(label)
        imgdraw = ImageDraw.Draw(img)

        list = []  # [0][0] => first hu moment of number zero
        with open('values') as fp:
            for line in fp:
                list.append(line[1:len(line) - 2].split(", "))

        # file = open("values", "w")

        for i in range(len(rects)):
            #   temp = Image.new('L', (200,200))
            imgdraw.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
            charimg = new_img.crop((rects[i][1] - 1, rects[i][0] - 1, rects[i][3] + 2, rects[i][2] + 2)).resize(
                (50, 50))
            #   temp.paste(charimg, (50, 50))
            #   charimg = temp
            # charimg.show()
            characterstr = self.matchshapes(charimg, list)
            imgdraw.text(((rects[i][1] + rects[i][3]) / 2, rects[i][0] - 10), characterstr, fill="red")
            # file.write(str(momentsof(np.asarray(charimg))) + "\n")

        # file.close()

        img.show()  # first image with rects and text

        # sampleize
        # file = open("values", "w")
        # for i in characters:
        #    print(momentsof(np.asarray(i)))
        #    file.write(str(momentsof(np.asarray(i))) + "\n")
        # file.close()

    def to_greyscale(self, img):
        img = img.convert('RGB')
        img_gray = img.convert('L')
        a_bin = self.threshold(np.asarray(img_gray), 100, THRESHOLD, 0)
        im = Image.fromarray(a_bin)
        return im

    def storesample(self, sample, character):
        index = 0
        if character[0].isalpha(): #convert to lowercase later on
            index = int(ord(character) - 96)
        elif character[0].isdigit():
            index = int(character[0])

        momentsdb = np.load("momentsdb.npy")

        for i in range(len(momentsdb)):
            for j in range(DICTIONARY_SIZE):
                if not np.any(momentsdb[i][j]):
                    momentsdb[i][index] = self.momentsof(sample)
                    np.save("momentsdb", momentsdb)
                    return None

    def matchshapes(self, shape, samplearr):
        shapemoments = self.momentsof(np.asarray(shape))
        difflist = [0] * 10

        for i in range(DICTIONARY_SIZE):
            for j in range(7):
                difflist[i] += abs(float(samplearr[i][j]) - shapemoments[j])

        # print(difflist.index(min(difflist)))
        # print('\n'.join(map(str, difflist)))
        return str(difflist.index(min(difflist)))

    def np2PIL(self, im):
        img = Image.fromarray(im, 'RGB')
        return img

    def np2PIL_color(self, im):
        img = Image.fromarray(np.uint8(im))
        return img

    def threshold(self, im, T, LOW, HIGH):
        (nrows, ncols) = im.shape
        im_out = np.zeros(shape=im.shape)
        for i in range(nrows):
            for j in range(ncols):
                if abs(im[i][j]) < T:
                    im_out[i][j] = LOW
                else:
                    im_out[i][j] = HIGH
        return im_out

    def blob_coloring_8_connected(self, bim):
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
                if c == THRESHOLD:
                    min_label = min(label_u, label_l, label_ul, label_ur)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label:
                            self.update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label:
                            self.update_array(a, min_label, label_l)

                        if min_label != label_ul and label_ul != max_label:
                            self.update_array(a, min_label, label_ul)

                        if min_label != label_ur and label_ur != max_label:
                            self.update_array(a, min_label, label_ur)

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

                if bim[i][j] == THRESHOLD:
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

    def update_array(self, a, label1, label2):
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

    def rectangles(self, labelsarr):
        nrow = labelsarr.shape[0]
        ncol = labelsarr.shape[1]

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

    def momentsof(self, image):
        rows = image.shape[0]
        cols = image.shape[1]
        raw_moments = [[0, 0, 0, 0], [0, 0, 0], [0, 0], [0]]

        k = 4
        for i in range(k):
            for j in range(k):
                for x in range(rows):
                    for y in range(cols):
                        raw_moments[i][j] += pow(x, i) * pow(y, j) * image[x][y]
            k = k - 1

        central_moments = [[0, 0, 0, 0], [0, 0, 0], [0, 0], [0]]
        xbar = raw_moments[1][0] / raw_moments[0][0]
        ybar = raw_moments[0][1] / raw_moments[0][0]

        k = 4
        for i in range(k):
            for j in range(k):
                for x in range(rows):
                    for y in range(cols):
                        central_moments[i][j] += pow(x - xbar, i) * pow(y - ybar, j) * image[x][y]
            k = k - 1

        scale_invariants = [[0, 0, 0, 0], [0, 0, 0], [0, 0], [0]]

        k = 4
        for i in range(k):
            for j in range(k):
                scale_invariants[i][j] = central_moments[i][j] / pow(central_moments[0][0], 1 + (i + j) / 2)
            k = k - 1

        rotation_invariants = [0, 0, 0, 0, 0, 0, 0]

        # indices indicate I sub index+1
        rotation_invariants[0] = scale_invariants[2][0] + scale_invariants[0][2]
        rotation_invariants[1] = pow(scale_invariants[2][0] - scale_invariants[0][2], 2) + 4 * pow(
            scale_invariants[1][1], 2)
        rotation_invariants[2] = pow(scale_invariants[3][0] - 3 * scale_invariants[1][2], 2) + pow(
            3 * scale_invariants[2][1] - scale_invariants[0][3], 2)
        rotation_invariants[3] = pow(scale_invariants[3][0] + scale_invariants[1][2], 2) + pow(
            scale_invariants[2][1] + scale_invariants[0][3], 2)
        rotation_invariants[4] = (scale_invariants[3][0] - 3 * scale_invariants[1][2]) * (
                scale_invariants[3][0] + scale_invariants[1][2]) * (
                                         pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - 3 * pow(
                                     scale_invariants[2][1] + scale_invariants[0][3], 2)) + (
                                         3 * scale_invariants[2][1] - scale_invariants[0][3]) * (
                                         scale_invariants[2][1] + scale_invariants[0][3]) * (
                                         3 * pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - pow(
                                     scale_invariants[2][1] + scale_invariants[0][3], 2))
        rotation_invariants[5] = (scale_invariants[2][0] - scale_invariants[0][2]) * (
                pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - pow(
            scale_invariants[2][1] + scale_invariants[0][3], 2)) + 4 * scale_invariants[1][1] * (
                                         scale_invariants[3][0] + scale_invariants[1][2]) * (
                                         scale_invariants[2][1] + scale_invariants[0][3])
        rotation_invariants[6] = (3 * scale_invariants[2][1] - scale_invariants[0][3]) * (
                scale_invariants[3][0] + scale_invariants[1][2]) * (
                                         pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - 3 * pow(
                                     scale_invariants[2][1] + scale_invariants[0][3], 2)) - (
                                         scale_invariants[3][0] - 3 * scale_invariants[1][2]) * (
                                         scale_invariants[2][1] + scale_invariants[0][3]) * (
                                         3 * pow(scale_invariants[3][0] + scale_invariants[1][2], 2) - pow(
                                     scale_invariants[2][1] + scale_invariants[0][3], 2))

        # print(cv2.HuMoments(cv2.moments(image)))
        # print(rotation_invariants)

        for i in range(7):
            rotation_invariants[i] = np.log10(abs(float(rotation_invariants[i])))

        return rotation_invariants

    def getsamplecount(self, character):
        count = 0
        index = 0
        if character[0].isalpha(): #convert to lowercase later on
            index = int(ord(character) - 96)
        elif character[0].isdigit():
            index = int(character[0])

        momentsdb = np.load("momentsdb.npy")

        for i in range(len(momentsdb)):
            if np.any(momentsdb[i][index]):
                count = count+1

        return count
