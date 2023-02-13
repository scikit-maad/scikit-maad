from tests._paths import DATA_PATH
from example_gallery.basic.plot_audio_representation import \
    plot_audio_representation


def test_plot_audio_representation():
    # Arrange
    filename = str(DATA_PATH / 'spinetail.wav')

    # Act
    plot_audio_representation(filename=filename)

    # Assert
    # If we get here, we count it as success
    assert True
