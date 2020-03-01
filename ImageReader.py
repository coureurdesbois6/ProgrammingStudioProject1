from PIL import Image, ImageDraw
import numpy as np

DICTIONARY_SIZE = 36 #36 if alphabet is included, #10 if only numbers
THRESHOLD = 150
SAMPLE_SHAPE = (25,25)
HU = "momentsdb.npy"
R = "momentsdbr.npy"
ZERNIKE = "momentsdbz.npy"

class ImageReader:
    #Detects and labels characters in given image file
    def launch(self, dest, methoddb=HU):
        img = Image.open(dest)
        im = self.to_greyscale(img) #greyscale image
        a_bin = np.asarray(im)
        label = self.blob_coloring_8_connected(a_bin)
        new_img = Image.fromarray(np.uint8(a_bin))
        rects = self.rectangles(label)
        imgdraw = ImageDraw.Draw(img)
        chars = []

        for i in range(len(rects)):
            imgdraw.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
            charimg = new_img.crop((rects[i][1] - 1, rects[i][0] - 1, rects[i][3] + 2, rects[i][2] + 2)).resize(
                SAMPLE_SHAPE)
            characterstr = self.matchshapes(charimg, methoddb=methoddb)
            chars.append(characterstr)
            imgdraw.text(((rects[i][1] + rects[i][3]) / 2, rects[i][0] - 10), characterstr, fill="red")

        with open('output.txt', 'w') as file:
            file.write(dest + "\n")
            if methoddb == HU:
                file.write("method: Hu moments\n")
            elif methoddb == R:
                file.write("method: R moments\n")
            elif methoddb == ZERNIKE:
                file.write("method: Zernike moments\n")
            file.write("Detected " + str(len(rects)) + " characters\n")
            for i in range(len(rects)):
                file.write(str(rects[i]) + " = " + str(chars[i]) + "\n")


        img.show()  # first image with rectangles and text

    #converts given image to greyscale image
    def to_greyscale(self, img):
        img = img.convert('RGB')
        img_gray = img.convert('L')
        a_bin = self.threshold(np.asarray(img_gray), 100, THRESHOLD, 0)
        im = Image.fromarray(a_bin)
        return im

    #maps given character to a number:
    #0-9 -> 0-9
    #A-Z -> 10-35
    def char_to_index(self, character):
        index = 0
        if character[0].isalpha():
            index = int(ord(character.lower()) - 87)
        elif character[0].isdigit():
            index = int(character[0])

        return index

    #Reverse map char_to_index()
    def index_to_char(self, index):
        char = 0
        if index >= 10:
            char = chr(int(index + 55))
        elif index < 10:
            char = index

        return str(char)

    #Store sample image to it's corresponding database file
    def storesample(self, sample, character, momentsdb=None, methoddb=HU):
        index = self.char_to_index(character)

        func = 0
        if momentsdb is None:
            try:
                momentsdb = np.load(methoddb)
            except:
                size = 7
                if methoddb == HU:
                    size = 7
                elif methoddb == R:
                    size = 10
                elif methoddb == ZERNIKE:
                    size = 12
                momentsdb = np.zeros(shape=(1, 36, size), dtype=float)
                np.save(methoddb, momentsdb)

        if methoddb == HU:
            func = self.hu_moments(sample)
        elif methoddb == R:
            func = self.r_moments(self.hu_moments(sample))
        elif methoddb == ZERNIKE:
            func = self.zernike_moments(sample)

        for i in range(len(momentsdb)):
            for j in range(DICTIONARY_SIZE):
                if not np.any(momentsdb[i][j]):
                    momentsdb[i][index] = func
                    print(momentsdb[i][index], i, index)
                    np.save(methoddb, momentsdb)
                    return None

        size = 7
        if methoddb == HU:
            size = 7
        elif methoddb == R:
            size = 10
        elif methoddb == ZERNIKE:
            size = 12

        newdim = np.zeros(shape=(DICTIONARY_SIZE, size), dtype=float)
        momentsdb = np.vstack((momentsdb, newdim[None]))
        self.storesample(sample, character, momentsdb=momentsdb, methoddb=methoddb)

    #Match given image to one of the shapes in the database
    def matchshapes(self, shape, methoddb=HU):
        momentsdb = np.load(methoddb)
        func = 0
        size = 7
        if methoddb == HU:
            func = self.hu_moments(np.asarray(shape))
            size = 7
        elif methoddb == R:
            func = self.r_moments(self.hu_moments(np.asarray(shape)))
            size = 10
        elif methoddb == ZERNIKE:
            func = self.zernike_moments(np.asarray(shape))
            size = 12
        shapemoments = func
        difflist = [0] * DICTIONARY_SIZE

        for i in range(len(momentsdb)):
            for j in range(DICTIONARY_SIZE):
                numofsamples = self.getsamplecount(self.index_to_char(int(j)), methoddb=methoddb)
                for k in range(size):
                    difflist[j] += abs(float(momentsdb[i][j][k] - shapemoments[k])) / numofsamples

        return str(self.index_to_char(difflist.index(min(difflist))))

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

    #Blob coloring algorithm to differantiate between distinct shapes/characters
    #in a given binary image
    def blob_coloring_8_connected(self, bim):
        max_label = int(10000)
        nrow = bim.shape[0]
        ncol = bim.shape[1]
        im = np.zeros(shape=(nrow, ncol), dtype=int)
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
        for i in range(k + 1):
            index = i
            while a[index] != index:
                index = a[index]
            a[i] = a[index]

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
            else:
                break

        return

    #Gives the (x0, y0, x1, y1) rectangle coordinates surrounding distinct shapes in a given labels array
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

        if rectangles[0][2] == nrow-1 and rectangles[0][3] == ncol-1:
            rectangles = np.delete(rectangles, 0, 0)

        rectangles = sorted(rectangles, key=lambda item: ((item[2] - item[0])/2, (item[3] - item[1])/2))
        #TODO: Sort rectangles from left to right, top to bottom
        #rects[i][1], rects[i][0], rects[i][3], rects[i][2]
        return rectangles

    #Calulate hu moments of a given image
    def hu_moments(self, image):
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

    #Calculate R moments of a given hu moments array.
    def r_moments(self, hu_moments):
        r_moments = [] * 10
        r_moments[0] = np.sqrt(hu_moments[1]) / hu_moments[0]
        r_moments[1] = (hu_moments[0] + np.sqrt(hu_moments[1])) / (hu_moments[0] - np.sqrt(hu_moments[1]))
        r_moments[2] = np.sqrt(hu_moments[2]) / np.sqrt(hu_moments[3])
        r_moments[3] = np.sqrt(hu_moments[2]) / np.sqrt(abs(hu_moments[4]))
        r_moments[4] = np.sqrt(hu_moments[3]) / np.sqrt(abs(hu_moments[4]))
        r_moments[5] = abs(hu_moments[5]) / (hu_moments[0] * hu_moments[2])
        r_moments[6] = abs(hu_moments[5]) / (hu_moments[0] * np.sqrt(abs(hu_moments[4])))
        r_moments[7] = abs(hu_moments[5]) / (hu_moments[2] * np.sqrt(hu_moments[1]))
        r_moments[8] = abs(hu_moments[5]) / (np.sqrt(hu_moments[1] * abs(hu_moments[4])))
        r_moments[9] = abs(hu_moments[4]) / (hu_moments[2] * hu_moments[3])

        return r_moments

    #Get the number of samples of a stored character in the database
    def getsamplecount(self, character, methoddb=HU):
        count = 0
        index = self.char_to_index(character)

        momentsdb = np.load(methoddb)

        for i in range(len(momentsdb)):
            if np.any(momentsdb[i][index]):
                count = count+1

        return count

    #TODO: Calculate zernike moments
    def zernike_moments(self, param):
        return 0
