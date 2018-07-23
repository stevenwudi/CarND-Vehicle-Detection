"""
Python class that handles vehicle detection. It has two modes
Full Scan: This is during initialization when all of the lanes and all positions, 224 in a four-lane highway,
are used in the sliding window before Voxel Occlusion constraint propagation technique is applied.

Sentinel Scan: This is for video after full scan is complete.
Only entry points in the lane lines are now scanned; and thus, drastically reduce number of sliding window searchs per
frame from 224 to just 9 for a four lane highway even before applying Voxel Occlusion constraint propagation.
"""
import os
import time

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from lib.roadGrid import RoadGrid
from skimage.feature import hog
from lib.maskRCNN import maskRCNN


# a class for wrapping our SVM trained HOG vehicle detector.
class VehicleDetection:
    # initialize
    def __init__(self, projectedX, projectedY,
                 maskRCNN_threshold_occupancy=0.5):
        self.start = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        self.projectedX = projectedX
        self.projectedY = projectedY
        # Using mask RCNN for vehicle detection
        self.maskRCNN = maskRCNN()
        self.maskRCNN_threshold_occupancy = maskRCNN_threshold_occupancy

    # Define a function to change the detector's threshold
    def set_threshold(self, new_threshold):
        self.maskRCNN_threshold_occupancy = new_threshold

    # Define a function to compute binned color features
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram( img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(
                img, orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True,
                visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(
                img, orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True, visualise=vis,
                feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, image, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256), orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):

        if image.shape[0] > 0 and image.shape[1] > 0:
            if image.shape[0] != 64 or image.shape[1] != 64:
                image = cv2.resize(image, (64, 64))

            # Create a list to append feature vectors to
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'GRAY':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif cspace == 'GRAYRGB':
                    rgbfeature_image = np.copy(image)
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                feature_image = np.copy(image)
            # Apply bin_spatial() to get spatial color features
            if cspace == 'GRAYRGB':
                spatial_features = self.bin_spatial(
                    rgbfeature_image, size=spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(
                    rgbfeature_image, nbins=hist_bins,
                    bins_range=hist_range)
                # Call get_hog_features() with vis=False, feature_vec=True
                hog_features = self.get_hog_features(
                    feature_image, orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                hogFeatures = np.concatenate(
                    (spatial_features, hist_features, hog_features))
            elif cspace == 'GRAY':
                hog_features = self.get_hog_features(
                    feature_image, orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True)
                hogFeatures = hog_features
            else:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
                # Call get_hog_features() with vis=False, feature_vec=True
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell,
                                                     cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                hogFeatures = np.concatenate((spatial_features, hist_features, hog_features))
            return self.X_scaler.transform(hogFeatures.reshape(1, -1))
        else:
            return None

    def slidingWindows(self, lines, laneIdx, complete=False):
        """
        Specialized sliding window generation. we are looking at top down birds-eye view and limiting the detection to
        just the lanes. We need to use the lane lines to help generate the sliding window locations.
        :param lines:
        :param laneIdx:
        :param complete:
        :return:
        """
        # calculate the window positions
        nlanes = len(lines) - 1
        x0 = self.projectedX / 2
        y0 = self.projectedY

        # create roadgrid for boxes
        window_list = RoadGrid(x0, y0, nlanes, laneIdx)

        for i in range(nlanes):
            lane_boxes = {}
            leftPolynomial = np.poly1d(lines[i].currentFit)
            rightPolynomial = np.poly1d(lines[i + 1].currentFit)

            # horizontal lines
            # we treat left and right lanes differently because of the
            # projection.  In the 'complete' case we are getting all
            # of the sliding windows
            if complete:
                if i < laneIdx:
                    indexedBottom = i + 1
                else:
                    indexedBottom = i
                for j in range(int(lines[indexedBottom].bottomProjectedY / 32)):
                    y1 = 32 * j
                    mid = int((rightPolynomial([y1]) + leftPolynomial([y1])) / 2)
                    x1 = mid - 32
                    x2 = mid + 32
                    y2 = y1 + 64
                    if x1 > 0 and x2 < self.projectedX and y1 > 0 and y2 < self.projectedY:
                        lane_boxes['%d' % j] = ((x1, y1), (x2, y2))

            # In the else case we are getting only the windows at the top
            # and bottom of our lanes for the sliding windows
            else:
                linetop = lines[i].getTopPoint()
                if i == laneIdx:
                    ylist = [(linetop[1], 0),
                             (linetop[1] + 32, 1),
                             (linetop[1] + 64, 2)]
                elif i < laneIdx:
                    ylist = [(linetop[1], 0),
                             (linetop[1] + 32, 1),
                             (linetop[1] + 64, 2),
                             (lines[i].bottomProjectedY - 96, 55)]
                else:
                    ylist = [(linetop[1], 0),
                             (linetop[1] + 32, 1),
                             (linetop[1] + 64, 2),
                             (lines[i + 1].bottomProjectedY - 32, 55)]

                for y1, j in ylist:
                    mid = int((rightPolynomial([y1]) + leftPolynomial([y1])) / 2)
                    x1 = mid - 32
                    x2 = mid + 32
                    y2 = y1 + 64
                    if x1 > 0 and x2 < self.projectedX and y1 > 0 and y2 < self.projectedY:
                        lane_boxes['%d' % j] = ((x1, y1), (x2, y2))
            window_list.map_boxes(i, lane_boxes)
        return window_list

    # draw_boxes function
    def draw_boxes(self, img, windows, color=(255, 255, 255), thick=20):
        # Iterate through the bounding boxes in a windows list
        for bbox in windows:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(
                img, (int(bbox[0][0]), int(bbox[0][1])),
                (int(bbox[1][0]), int(bbox[1][1])), color, thick)

    # Define a way for us to write out a sample of the HOG
    def drawPlots(self, imagefile, sampleTitle, images):
        # print("saving image and hog results to ", imagefile)
        # Setup plot
        fig = plt.figure(figsize=(12, len(images) * 9))
        w_ratios = [2.0, 6.5, 6.5]
        h_ratios = [9.0 for n in range(len(images))]
        grid = gridspec.GridSpec(
            len(images), 3, wspace=0.05, hspace=0.0,
            width_ratios=w_ratios, height_ratios=h_ratios)
        i = 0

        for filename, orient, pix_per_cell, \
            cell_per_block, image1, image2 in images:
            # draw the images
            # next image
            title = '%s\n Orientation: %d\n'
            title += ' Pix_per_cell: %d\n'
            title += ' Cell_per_block: %d'
            title = title % \
                    (filename, orient, pix_per_cell, cell_per_block)

            ax = plt.Subplot(fig, grid[i])
            ax.text(-0.5, 0.4, title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image1)
            if i == 1:
                ax.set_title('Original', size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

            ax = plt.Subplot(fig, grid[i])
            ax.imshow(image2)
            if i == 2:
                ax.set_title('Augmented %s' % (sampleTitle), size=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            i += 1

        plt.savefig(imagefile)
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        y, x, ch = image.shape
        cuttoff = int((y / len(images)) * 0.65)
        image = image[cuttoff:(y - cuttoff), :, :]
        cv2.imwrite(imagefile, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Define a way for us to process an image with
    # a list of sliding windows and try to detect vehicles
    def detectVehicles(self, image, roadgrid):
        mapping = roadgrid.getMapping()
        for box in mapping.keys():
            if not mapping[box]['occluded'] and not mapping[box]['found'] and mapping[box]['vehicle'] is None:
                window = mapping[box]['window']
                wimage = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
                wfeatures = self.extract_features(
                    wimage, cspace=self.cspace, spatial_size=(32, 32),
                    orient=self.orient, pix_per_cell=self.pix_per_cell,
                    cell_per_block=self.cell_per_block,
                    hog_channel=self.hog_channel,
                    hist_bins=32, hist_range=(0, 256))
                if wfeatures is not None:
                    confidence = self.svc.decision_function(wfeatures.reshape(1, -1))
                    print(confidence[0])
                    if confidence[0] > self.threshold:
                        roadgrid.setFound(box)
        return roadgrid

    def detectVehiclesDNN(self, image):
        """
        This is a Deep Neural netowkr based vehicle detection
        :param image:
        :param roadgrid:
        :return:
        """
        cls_boxes, cls_segms, prediction_row = self.maskRCNN.vehicleDetection(image)
        binary_mask, scores, instance_id = self.maskRCNN.binary_mask(cls_boxes, cls_segms)
        if not len(instance_id):
            binary_mask = np.zeros_like(image)
            scores = []
            instance_id = []
        return cls_boxes, cls_segms, binary_mask, scores, instance_id

    def assignVehiclesRoadGrid(self, mask_all, roadgrid, instance_id):
        mapping = roadgrid.getMapping()
        mapping_keys = [n for n in mapping.keys()]
        mapping_keys.sort()

        for i_id in instance_id:
            binary_mask = (mask_all % 100 / 10).astype(int) == i_id
            for box in mapping_keys:
                if not mapping[box]['occluded'] and not mapping[box]['found'] and mapping[box]['vehicle'] is None:
                    window = mapping[box]['window']
                    wimage = binary_mask[window[0][1]:window[1][1], window[0][0]:window[1][0]]
                    # For mask RCNN prediction: int(instance_count/10) is instance number
                    confidence = np.mean(wimage)
                    if confidence > self.maskRCNN_threshold_occupancy:
                        print(confidence, i_id, box)
                        roadgrid.setFound(box, i_id, confidence)
                        # break
        return roadgrid

    # Define a way for us to collect data from images and videos
    def collectData(self, frame, image, windows):
        baseDir = "collected/%s/%04d/" % (self.start, frame)
        if not os.path.exists(baseDir):
            os.makedirs(baseDir)
        i = 0
        for window in [lane for lane in windows]:
            wimage = image[window[0][1]:window[
                1][1], window[0][0]:window[1][0]]
            outfilename = baseDir + self.dataFileNamePattern % (i)
            cv2.imwrite(outfilename,
                        cv2.cvtColor(wimage, cv2.COLOR_RGB2BGR))
            i += 1
