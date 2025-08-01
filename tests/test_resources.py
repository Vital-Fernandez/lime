import pytest
from lime.resources.generator_logo import lime_log_function

tolerance_rms = 5.5

@pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
def test_logo():

    fig = lime_log_function(None, show=False, style='default')

    return fig
