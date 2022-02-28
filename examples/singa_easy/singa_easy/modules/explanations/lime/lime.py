from lime import lime_image

from skimage.segmentation import mark_boundaries
from singa_auto.model import utils


class Lime:
    """
    Lime: Explaining the predictions of any machine learning classifier
    https://github.com/marcotcr/lime
    """

    def __init__(self, model,
                 image_size,
                 normalize_mean,
                 normalize_std,
                 device):

        self._model = model
        self.device = device
        # dataset
        self._image_size = image_size
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self._explainer = lime_image.LimeImageExplainer()
        # lime configs
        # number of images that will be sent to classification function
        self._num_samples = 100
        self._top_labels = 5
        self._hide_color = 0

    def batch_predict(self, images):
        (images, _, _) = utils.dataset.normalize_images(images,
                                                        self._normalize_mean,
                                                        self._normalize_std)

        self._model.eval()

        images = images.to(self.device)
        logits = self._model(images).to(self.device)

        return probs.detach().cpu().numpy()

    def explain(self, images):
        img_boundry = []
        for img in images:
            explanation = self._explainer.explain_instance(
                img, self.batch_predict, self._top_labels, self._hide_color,
                self._num_samples)
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False)
            # (M, N, 3) array of float
            img_boundry = mark_boundaries(temp / 255.0, mask)
        return img_boundry * 255
