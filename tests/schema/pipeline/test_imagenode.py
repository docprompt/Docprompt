from docprompt.schema.pipeline import ImageNode


def test_imagenode():
    ImageNode(image=b"test", metadata={})
