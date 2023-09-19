from specarray import SpecArray


def test_from_dir(testdata_specim_folder):
    spec = SpecArray.from_folder(testdata_specim_folder)
    print(spec)
