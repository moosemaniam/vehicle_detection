def feat_extractor_img(img, color_space='RGB', spatial_size=(32, 32),
    hist_bins=32, orient=9,
    pix_per_cell=8, cell_per_block=2, hog_channel=0,
    spatial_feat=True, hist_feat=True, hog_feat=True):
  #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
      if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
      elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
      elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
      elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
      elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    if spatial_feat == True:
      spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    if hist_feat == True:
      hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
      if hog_channel == 'ALL':
        hog_features = []
            for channel in range(feature_image.shape[2]):
              hog_features.extend(get_hog_features(feature_image[:,:,channel],
                orient, pix_per_cell, cell_per_block,
                vis=False, feature_vec=True))
        else:
          hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
              pix_per_cell, cell_per_block, vis=False, feature_vec=True)
          #8) Append features to list
        img_features.append(hog_features)
    #9) Return concatenated array of features
    return np.concatenate(img_features)

def window_search(img,windows,
    color_space,
    orient,
    pix_per_cell,
    cell_per_block,
    hog_channel,
    spatial_size,
    hist_bins,
    spatial_features,
    hist_features,
    hog_features):
  """Searches all the input windows cars.
    Returns a list of windows which have cars"""
     positive_windows=[]
     for window in windows:
       #Get the pixel data for the window in question
        window_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

          #Extract features for this window
        features = feat_extractor_img(window_img,
            color_space,
            spatial_size,
            hist_bins,
            orient,
            pix_per_cell,
            cell_per_block,
            hog_channels,
            spatial_features,
            hist_features,
            hog_features)
        #Scale the features
        window_features = scaler.transform(np.array(features).reshape(1, -1))
        is_car = clf.predict(window_features)

        if(is_car == 1):
          positive_windows.append(window)

    return positive_windows
