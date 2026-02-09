import data


def main():
    ImageData = data.load_data()
    print(ImageData.images_train[0])


if __name__ == "__main__":
    main()
