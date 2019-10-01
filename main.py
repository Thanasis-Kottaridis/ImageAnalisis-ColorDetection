# opencv-python, opencv-contrib-python 3.4.2.16
# scikit-image
# numpy
# scipy
# matplotlib
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

def training(train_images, quant_colors, segments_num):

    kmap_a = [] # edo tha apothikeuti to a gia kathe train_image
    kmap_b = [] # edo tha apothikeuti to b gia kathe train_image

    dataset = [] # edo apothikeuete to dataset pou prokipti apo tin e3agogi xaraktiristikon apo tis eikones

    # gia kathe eikona tou sinolou ekpedeusis
    # diavazei tin original image tin metatrepei se Lab
    # diaxorizei ta a,b
    # kai enimeronei ta kmap_a, kmap_b

    for image in train_images:
        original_img = cv2.imread(image)
        Lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        _, a, b = cv2.split(Lab_img)
        kmap_a = np.concatenate([kmap_a, a.flatten()])
        kmap_b = np.concatenate([kmap_b, b.flatten()])

    # otan diavasei ola ta a kai b apo tis eikones ekpedeusis
    # ta enonei kai ta kanei enan disdiastato pinaka
    # ton opoion dinei san isodo ston kmeans kai autos ipologizei ta kentroidi (quant_colors)
    pixel = np.squeeze(cv2.merge((kmap_a.flatten(), kmap_b.flatten())))
    kmeans = MiniBatchKMeans(quant_colors)
    #kanei updateton kmeans
    kmeans.fit_predict(pixel)
    centroids = kmeans.cluster_centers_.astype("uint8")
    print(centroids)



    # gia kathe image tin diavazei kai tin metatrepei ston xromatiko xwro lab
    # kai sti sinexia tin 3anaxromatizei pixel pros pixel
    q_im = []
    for image in train_images:
        original_img = cv2.imread(image)
        (h, w) = original_img.shape[:2]
        Lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(Lab_img)
        pixel = np.reshape((cv2.merge((a, b))), (h*w, 2))
        # provlepei kathe pixel se poio cluster anikei
        # diladi o indexes_of_pixels periexei se poio cluster anikei to antistixo pixel
        indexes_of_pixels = kmeans.predict(pixel)
        #print(indexes_of_pixels)
        np.asarray(indexes_of_pixels)


        # xromatismos ton pixel me to kvantismeno xroma
        quant_a = np.empty((h, w)).astype("uint8")
        quant_b = np.empty((h, w)).astype("uint8")
        counter = 0
        for i in range(0, h):
            for j in range(0, w):
                quant_a[i, j] = centroids[indexes_of_pixels[counter], 0]
                quant_b[i, j] = centroids[indexes_of_pixels[counter], 1]
                counter = counter + 1

        quant_img = cv2.merge((l, quant_a, quant_b))
        # cv2.imshow("Original image", original_img)
        # cv2.imshow("Lab_image", Lab_img)
        # cv2.imshow("quant_image", quant_img)
        # cv2.imshow("quant2_image", cv2.cvtColor(quant_img, cv2.COLOR_LAB2BGR))
        # cv2.waitKey(0)

        """
            efarmozw SLIC kai e3agw (peripou) 
            ton arithmo ton segments pou eisagame
        """
        segments = slic_superpixels(quant_img, segments_num, 0.1)

        # gia kathe ena apo ta segments pragmattopoiounte ta eksis
        # 1) ipologismos tou epikratesterou xromatos
        # 2) eksagogi xaraktiristikon ifis meso tou algorithmou surf
        # 3) eksagogi xaraktiristikon ifis meso tou algorithmou gabor

        for (i, segVal) in enumerate(np.unique(segments)):
            # dimiorgoume mia maska gia to segment
            print("[x] inspecting segment {}".format(i))
            mask = np.zeros(quant_img.shape[:2], dtype="uint8")
            mask[segments == segVal] = 255
            superpixel = cv2.bitwise_and(quant_img, quant_img, mask=mask)

            # show the masked region
            # cv2.imshow("Mask", mask)
            # cv2.imshow("Applied", superpixel)
            # cv2.waitKey(0)

            # gia kathe ena apo ta segments pragmattopoiounte ta eksis
            # 1) ipologismos tou epikratesterou xromatos
            # 2) eksagogi xaraktiristikon ifis meso tou algorithmou surf
            # 3) eksagogi xaraktiristikon ifis meso tou algorithmou gabor

            # gia kathe superpixel vriskei to kiriarxo xroma
            # vriskei prota to xroma tou kathe pixel pou anikei sto superpixel
            superpixel_pixels_color = np.vstack(superpixel[segments == segVal, 1:])
            superpixel_predominant_color = get_superpixel_predominant_color(superpixel_pixels_color, kmeans)
            print("DOMINANT COLOR {}".format(superpixel_predominant_color))

            # ipologismos xaraktiristikon ifis gabor gia to L(ta xaraktiristika tis fotinotitas) tou superpixel
            superpixel_gabor_features = np.hstack(get_gabor_features(superpixel[:, :, 0]))
            print("GABOR {}".format(len(superpixel_gabor_features)))
            superpixel_surf_features = get_surf_features(quant_img[:, :, 0], np.argwhere(segments == segVal),
                                                         n_keypoints=7, surf_window=20)

            # ta prosthetei ola stin lista me ta xaraktiristika
            dataset.append(
                np.hstack((superpixel_surf_features, superpixel_gabor_features, superpixel_predominant_color)))
            print("-----------------------------------------------------------")

    result = pd.DataFrame(dataset)
    return train_svm(result), centroids


'''
    methodos gia katatmisi tis eikonas se superpixels
    ta superpixel einai omadopoiimena pixels vasi koinon xaraktiristikon
    (color kai texture) einai poio apodotika apo ta apla pixel giati 
    periexoun perisoteri pliroforia kai poio apotelesmatika ipologistika
    kai poio eukola stin diaxirisi tous mesa se grafimata
    ---Dexete san parametrous tin eikona kai ta segments pou tha tin xorisei---
'''


def slic_superpixels(image, numSegments, compactness= 0.075):
    # diavazei tin eikona kai tin metatrepei se floatin point data type
    image = img_as_float(image)

    # efarmozei slic stin eikona kai e3agei kataprosegkisi ton arithmo ton segments
    segments = slic(image, n_segments=numSegments, sigma=5, compactness=compactness)

    # # kodikas gia na dixnei ta supepixels
    # fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments))
    # s = mark_boundaries(image, segments)
    # plt.axis("off")
    #
    # # show the plots
    # plt.show()
    return segments


'''
    methodos gia ipologismo tou kiriarxou xromatos ana superpixel
    dexete enna disdiastato array to opoio periexei to a kai to b 
    gia kathe pixel tou superpixel kai stisinexia efarmozei kmeans 
    gia na dei poio kentroidi anaparista to sigkekriemeno xroma
    sti sinexeia metraei poio kentroidi emfanizete perisoteres 
    fores kai to epistreuei
'''


def get_superpixel_predominant_color(superpixel_pixel_color, kmeans):
    indexes_of_pixels = kmeans.predict(superpixel_pixel_color)
    count = np.bincount(indexes_of_pixels)
    perdominant_color = np.argmax(count)
    return perdominant_color


def get_surf_features(img, superpixel, hessianThreshold=400, nOctaves=3, n_keypoints=10, surf_window=20):

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves)
    key_points = [cv2.KeyPoint(y, x, surf_window) for (x, y) in superpixel]
    key_points = sorted(key_points, key=lambda x: -x.response)[:n_keypoints]
    kp, descritors = surf.compute(img, key_points)
    if len(kp) != 0:
        des = descritors.flatten()
    else:
        des = np.array([])

    feature_vector_size = (n_keypoints * 64)

    if descritors is not None:
        if descritors.size < feature_vector_size:
            des = np.concatenate([des, np.zeros(feature_vector_size - descritors.size)])

    else:
        des = np.zeros(feature_vector_size)
        print("LENGTH OF DESCRIPTOR NONE")
    return des


'''
    ipologizei ta gabor features kai ta efarmozei sto superpixel epistrefontas ena Response Matrices
    kai sti sinexeia ipologizei kai epistrefei to local energy kai to mean amplitude tou superpixel
    local_energy = athrizei to tetragono ton stixion tou Response Matrix
    mean_amplitude = athrizei tis apolites times ton stixion tou Response Matrix
'''


def get_gabor_features(img):
    local_energy = []
    mean_amplitude = []
    ksize = 20
    theta_range = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 4, 5 * np.pi / 6]
    scale_range = [3, 6, 13, 28, 58]
    # ftiaxnei ta filtra tou gabor (gabor kernels)
    # kai ta efarmozei stin img
    for scale in scale_range:
        for theta in theta_range:
            kern = cv2.getGaborKernel((ksize, ksize), scale, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            gabor_image = cv2.filter2D(img, cv2.CV_8UC3, kern)

            local_energy.extend([np.sum(np.square(gabor_image))])
            mean_amplitude.extend([np.sum(np.absolute(gabor_image))])

    return local_energy, mean_amplitude


def train_svm(dataset):
    # dimiourgei ton svm ta3inomiti
    # kai ton one vs all ta3inomiti o opoios ta3inomei to dataset vasi tou svm pou dimiourgisame
    # svm = SVC(kernel='rbf', gamma='scale', cache_size=700, random_state=110)
    # svm = OneVsRestClassifier(svm, n_jobs=-1)

    svm = LinearSVC(multi_class='ovr', max_iter=100000)

    # diaxorizei ta features apo tis klasis
    X = dataset.drop(dataset.columns[(64*7)+80], axis=1)
    Y = dataset[dataset.columns[(64*7)+80]]

    # xorizei to dataset se 2 epimerous dataset (ekpedefsis kai test)
    # opou to test dataset einai to 20% tou arxikou dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # kanei fit to train dataset ston on vs all algorithmo me vrisi svm
    svm.fit(X, Y)

    # elenxei tin apotelesmatikotita tou svm
    # prediction = svm.predict(X_test)
    # print(prediction)
    # print("ACCURACY: {}".format(metrics.accuracy_score(y_test, prediction)))
    return svm


'''
    methodos gia xromatismo aspromavris eikonas dexete tin eikona stoxos (target_image)
    to ekpedevmeno SVM (trained_svm) pou proekipse apo tin ektelesi tis methodou ekpedefsis
    kai tin lista me ta kentroidi tou Kmeans pou apoteloun ta kvantismena xromata
    o xromatismos tis arpromavris eikonas ginete me simfona me ta akoloutha vimata
    1) metatropi tis eikonas stoxos ston xromatiko xoro Lab
    2) katatmisi tis eikonas stoxos se superpixel
    3) e3agogi xaraktiristikon ifis gia kathe superpixel apo tin eikona stoxos (surf, gabor features)
    4) provlepsi se pia klasi aniki to kathe superpixel
    5) xromatismos tou kathe superpixel
'''


def colorization(target_image, trained_svm, cendroids, segments):
    # diavazei tin eikona stoxos se rgb kai Lab
    original_img = cv2.imread(target_image)
    Lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    # diavazei tin eikona stoxos se gray
    target_img = Lab_img[:, :, 0]
    # arxikopoiei ena adio array me tis diastasis tis eikonas stoxo metasximatismeni ston xwro lab
    colorized_img = np.zeros_like(Lab_img)
    # vazei tin gray target img san L stin colorized_img
    colorized_img[:, :, 0] = target_img

    # diaspa tin eikona stoxos(target_img) se segments me xrisi tou algorithmou slic
    # segments = slic_superpixels(Lab_img, 600)
    segments = slic(target_img, n_segments=segments, sigma=5, compactness=0.2)

    # # kodikas gia na dixnei ta supepixels
    # fig = plt.figure("Superpixels -- %d segments" % (100))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(target_img, segments))
    # s = mark_boundaries(target_img, segments)
    # plt.axis("off")

    # show the plots
    # plt.show()

    features = []
    # gia kathe segment efarmozei mia mask gia na prokipsei to superpixel
    for (i, segVal) in enumerate(np.unique(segments)):

        # dimiorgoume mia maska gia to segment
        mask = np.zeros(target_img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        superpixel = cv2.bitwise_and(target_img, target_img, mask=mask)

        # se kathe superpixel e3agei xaraktiristika ifis meso ton algorithmo (gabor, surf)
        superpixel_gabor_features = np.hstack(get_gabor_features(superpixel))
        superpixel_surf_features = get_surf_features(np.uint8(target_img), np.argwhere(segments == segVal),
                                                     n_keypoints=7, surf_window=20)
        features.append(
            np.hstack((superpixel_surf_features, superpixel_gabor_features)))
        features1 = np.hstack((superpixel_surf_features, superpixel_gabor_features))

        prediction = trained_svm.predict(features1.reshape(1, -1))

        predict_a = centroids[int(prediction)][0]
        predict_b = centroids[int(prediction)][1]

        colorized_img[segments == segVal, 1:3] = np.array([predict_a, predict_b])
        #print(prediction)

    # prediction = trained_svm.predict(features)[0]
    # print(prediction)
    cv2.imshow("Original Image", original_img)
    cv2.imshow("Target Image", target_img)
    cv2.imshow("Colorized Image", cv2.cvtColor(colorized_img, cv2.COLOR_LAB2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# project main
if __name__ == "__main__":
    quant_colors = 4
    segments = 100
    trained_svm, centroids = training(['2.jpg', '2.jpg'], quant_colors, segments)
    colorization("4.jpg", trained_svm, centroids, segments)
