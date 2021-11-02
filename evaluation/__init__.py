import faiss, matplotlib.pyplot as plt, os, numpy as np, torch
from PIL import Image
import time
from tqdm import tqdm
from evaluation.normalized_mutual_information import calc_normalized_mutual_information, cluster_by_kmeans
from evaluation.recall import calc_recall_at_k, assign_by_euclidian_at_k

def predict_batchwise(model, dataloader):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    # print(list(model.parameters())[0].device)
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    _, (J, _) = model(J)
                    J = J.cpu()
                for j in J:
                    #if i == 1: print(j)
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]


def evaluate(model, dataloader, eval_nmi=True, recall_list=[1, 2, 4, 8]):
    '''
        Evaluation on dataloader
        :param model: embedding model
        :param dataloader: dataloader
        :param eval_nmi: evaluate NMI (Mutual information between clustering on embedding and the gt class labels) or not
        :param recall_list: recall@K
    '''
    eval_time = time.time()
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dataloader)

    print('done collecting prediction')

    if eval_nmi:
        # calculate NMI with kmeans clustering
        nmi = calc_normalized_mutual_information(
            T,
            cluster_by_kmeans(
                X, nb_classes
            )
        )
    else:
        nmi = 1

    print("NMI: {:.3f}".format(nmi * 100))

    # get predictions by assigning nearest 8 neighbors with euclidian
    max_dist = max(recall_list)
    Y = assign_by_euclidian_at_k(X, T, max_dist)
    Y = torch.from_numpy(Y)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in recall_list:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    chmean = (2 * nmi * recall[0]) / (nmi + recall[0])
    print("hmean: %s", str(chmean))

    eval_time = time.time() - eval_time
    print('Eval time: %.2f' % eval_time)
    return nmi, recall


def set_checkpoint(model, opt, progress_saver, savepath, aux=None):
    if 'experiment' in vars(opt):
        import argparse
        save_opt = {key:item for key,item in vars(opt).items() if key!='experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    torch.save({'state_dict':model.state_dict(), 'opt':save_opt, 'progress':progress_saver, 'aux':aux}, savepath)




##########################
def recover_closest_standard(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest+1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()
