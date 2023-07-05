import cupy as cp
import cv2

def convert_RGB2OD(img):
    """Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    Args:
        img (ndarray uint8): Image RGB
    Returns:
        ndarray: Optical denisty RGB image.
    """
    mask = img == 0
    img[mask] = 1
    return cp.maximum(-1 * cp.log(img / 255), 1e-6)

def convert_OD2RGB(OD):
    """Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)
    Args:
        OD (ndrray): Optical denisty RGB image
    Returns:
        ndarray uint8: Image RGB
    """
    OD = cp.maximum(OD, 1e-6)
    return (255 * cp.exp(-1 * OD)).astype(cp.uint8)

class StainNormaliser:
    """Stain normalisation base class.
    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Attributes:
        extractor (CustomExtractor,RuifrokExtractor): method specific stain extractor.
        stain_matrix_target (ndarray): stain matrix of target.
        target_concentrations (ndarray): stain concetnration matrix of target.
        maxC_target (ndarray): 99th percentile of each stain.
        stain_matrix_target_RGB (ndarray): target stain matrix in RGB.

    """

    def __init__(self):
        self.extractor = None
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

    @staticmethod
    def get_concentrations(img, stain_matrix):
        """Estimate concentration matrix given an image and stain matrix.
        Args:
            img (ndarray): input image.
            stain_matrix (ndarray): stain matrix for haematoxylin and eosin stains.
        Returns:
            ndarray: stain concentrations of input image.
        """
        OD = convert_RGB2OD(img).reshape((-1, 3))
        x, _, _, _ = cp.linalg.lstsq(stain_matrix.T, OD.T, rcond=-1)
        return x.T

    def fit(self, target):
        """Fit to a target image.
        Args:
            target (ndarray uint8): target/reference image.

        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(
            target, self.stain_matrix_target
        )
        self.maxC_target = cp.percentile(
            self.target_concentrations, 99, axis=0
        ).reshape((1, 2))
        # useful to visualize.
        self.stain_matrix_target_RGB = convert_OD2RGB(self.stain_matrix_target)

    def transform(self, img):
        """Transform an image.
        Args:
            img (ndarray uint8): RGB input source image.
        Returns:
            ndarray: RGB stain normalised image.
        """

        stain_matrix_source = self.extractor.get_stain_matrix(img)
        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        maxC_source = cp.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / maxC_source
        trans = 255 * cp.exp(
            -1 * cp.dot(source_concentrations, self.stain_matrix_target)
        )

        # ensure between 0 and 255
        trans[trans > 255] = 255
        trans[trans < 0] = 0

        return trans.reshape(img.shape).astype(cp.uint8)
class RuifrokExtractor:

    @staticmethod
    def get_stain_matrix(_):
        """Get the pre-defined stain matrix.
        Returns:
            ndarray: pre-defined  stain matrix.
        """
        return cp.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])

class RuifrokNormaliser(StainNormaliser):
  
    def __init__(self):
        super().__init__()
        self.extractor = RuifrokExtractor()


# norm = RuifrokNormaliser()
# norm.fit(target_image)
# norm_img = norm.transform(source_image)
