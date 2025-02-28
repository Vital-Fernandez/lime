from lime.inference.intensity_threshold import LineFinder

class FeatureDetection(LineFinder):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFinder.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        return