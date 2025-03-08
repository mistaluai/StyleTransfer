def gram_matrix(feature_maps):
    batch_size, num_channels, height, width = feature_maps.shape
    feature_maps = feature_maps.view(batch_size * num_channels, height * width)
    feature_maps_transposed = feature_maps.t()
    gram = feature_maps @ feature_maps_transposed
    gram_n = gram / (batch_size * num_channels * height * width)
    return gram_n
