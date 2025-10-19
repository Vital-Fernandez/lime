from lime.inference.intensity_threshold import LineFinder

try:
    from aspect.workflow import ComponentsDetector
    aspect_check = True
except ImportError:
    aspect_check = False


if not aspect_check:
    class FeatureDetection(LineFinder):
        def __init__(self, spectrum):

            # Instantiate the dependencies
            LineFinder.__init__(self)

            # Lime spectrum object with the scientific data
            self._spec = spectrum

            return

else:
    class FeatureDetection(LineFinder, ComponentsDetector):

        def __init__(self, spectrum):

            # Instantiate the dependencies
            LineFinder.__init__(self)
            ComponentsDetector.__init__(self, spectrum)

            # Lime spectrum object with the scientific data
            self._spec = spectrum

            return
