import torch

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_monte_carlo_predictions(feature,
                                forward_passes,
                                ad_net,
                                d_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes
    Parameters
    ----------
    inout_list : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    adnet : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
    ad_net.eval()
    enable_dropout(ad_net)
    dropout_predictions = torch.empty((0, n_samples, d_classes)).cuda()
    for _ in range(forward_passes):
        ad_out = ad_net(feature).cuda()
        dropout_predictions = torch.vstack((dropout_predictions, torch.unsqueeze(ad_out, 0))).cuda()

    # Calculating variance across multiple MCD forward passes
    variance = torch.var(dropout_predictions, dim=0)  # shape (n_samples, n_class

    return variance  

def get_mc_var(src_loader, tar_loader, base_network, ad_net, num_src_samples, num_tar_samples):
    base_network.eval()
    ad_net.eval()
    src_features_list = []
    tar_features_list = []
################################### get all features and softmax of src and tar ############################
    for p in iter(src_loader):
        with torch.no_grad():
            _, features= base_network(p[0].cuda())
        src_features_list.append(features)
    for q in iter(tar_loader):
        with torch.no_grad():
            _, features, = base_network(q[0].cuda())
        tar_features_list.append(features)
    src_features = torch.cat(src_features_list, 0)
    tar_features = torch.cat(tar_features_list, 0)
    del src_features_list
    del tar_features_list
################################## run 10 forward for mcdropout #####################################################
    src_variance = get_monte_carlo_predictions(src_features, 10, ad_net, 1,  num_src_samples)
    tar_variance = get_monte_carlo_predictions(tar_features, 10, ad_net, 1,  num_tar_samples)

################################### normalize all the variance #################################################
    src_transferbility = (src_variance - torch.min(src_variance)) / torch.max(src_variance)
    tar_transferbility = (tar_variance - torch.min(tar_variance)) / torch.max(tar_variance)
    return src_transferbility.detach(), tar_transferbility.detach()
