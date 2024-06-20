## Storage Design Pattern

### Seperation of Concerns

1. **Document Node**: The Document Node is repsonsibel for interacting with the storage layer and managing the loading of the metadata object conditionally.

    - The Document Node is responsible for exposing a `persist(self, path: Optional[str] = None, **kwargs)` method and a `from_storage(cls, path: str, file_hash: str, **kwargs)` method. This will be the developers primary method of interacting with the storage layer. There will be no need for the developer to have to instantiate their own storage provider class or file path models. All of the implementation with the storage layer is managed directly by the document ndoe.
    - For managing the metadata, the Document Node must be able to get the metadata model from the model fields. By extracting the metadata class from the model fields, the metadata model can be instantiated both from instance and class methods. This way, both the `persist` and `from_storage` method will be able to utilize metadata properly.

2. [x] **FSSpec Wrapper**: We want to create an wrapper around `fsspec` which provides an interface for reading, writing, and deleting file objects. Our wrapper will take two file paths, so that the document nodes can always be written as side cars.

    - We will want to provide appropriate `kwargs` support to allow users to implement any necessary functionality for reads and writes, such as encryption or compression, to ensure that a robust storage interface is provided for all application needs.

3. [x] **File Path Manager**: The last main component we need to implement is the file path manager. This path manager will handle taking a base path, as well as a file hash, and generating the paths for the pdf bytes and the sidecar bytes.
    - This model is responsible for path validation and path formatting.

### Interface Protocols

**Storing a Document Node**: When storing a document node, we will take the following steps:

1. Validate filesystem works (Instantiate FSSpec Wrapper)
2. Create a folder to store the PDF + sidecar at that Path, if it doesn't already exist (FSSpec Wrapper method called from document node)
3. Serialize the PDF and the sidecar to filesystem writable (Document node)
4. Write the PDF and the sidecar to the filesystem (Document Node using FSSpec wrapper)
5. Return the path to the PDF and the sidecar (Document Node)

**Reading a Document Node**: When loading a document node form storage, we will take the following steps:

1. Validate filesystem works (Instantiate FSSpec Wrapper)
2. Create file path manager for document node (Instantiate File Path Manager)
3. Validate that a folder exists in the base path for the file hash (FSSpec Wrapper method called from document node using FP manager)
4. Get the pydantic model for validating the metadata from Document Nodes model fields (Document Node)
5. Read bytes of sidecar files from the filesystem (Document node using FSSpec Wrapper)
6. Validate bytes as a PDF and metadata model (Document Node)
7. Return instantiated document node (Document Node)
