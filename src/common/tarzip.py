import tarfile
import os


class TarZip:

    """Tarzip compress an h5 model file"""

    @staticmethod
    def compress(file_name: str) -> None:
        """Tar compress a local file.

        :param file_name: name of local file to compress
        :type file_name: str
        """
        tar_file_name = os.path.splitext(file_name)[0]
        with tarfile.open(f"{tar_file_name}.tar.gz", "w:gz") as tar:
            tar.add(file_name)
        os.remove(file_name)

    def extract(file_name: str, directory: str) -> None:
        """Extract all files from tarfile

        :param file_name: name of local file to extract from
        :type file_name: str
        :param directory: name of directory to extract files into
        :type directory: str
        """
        with tarfile.open(file_name) as tar:
            for name in tar.getnames():
                member = tar.getmember(name)
                tar.extract(member, directory)


if __name__ == "__main__":
    pass
