from lime.inference.intensity_threshold import LineFinder
from aspect.workflow import ComponentsDetector

class FeatureDetection(LineFinder, ComponentsDetector):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        LineFinder.__init__(self)
        ComponentsDetector.__init__(self, spectrum)

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        return