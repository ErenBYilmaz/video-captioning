
from lib.util import EBC


def register_subclasses():
    """
    The actual registration takes place when the class is defined/imported.
    This is a side effect of the import statements at the top of the file
    Additionally, we check that the registration was successful.
    """
    import data.image_metadata_from_video
    import data.named_image_metadata
    assert data.named_image_metadata.NamedImageMetadata.__name__ in EBC.SUBCLASSES_BY_NAME
    assert data.image_metadata_from_video.ImageMetadataFromVideo.__name__ in EBC.SUBCLASSES_BY_NAME
