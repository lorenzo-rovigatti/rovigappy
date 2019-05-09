from datapackage import Package


AVAILABLE_DATA_LIST = ['english-premier-league', 'spanish-la-liga', 'italian-serie-a',
                       'german-bundesliga', 'french-ligue-1']


# TODO
def save_package_to_csv(package, name):
    pass


def main():

    for data_name in AVAILABLE_DATA_LIST:

        package = Package('https://datahub.io/sports-data/{}/datapackage.json'.format(data_name))

        # print list of all resources:
        print(package.resource_names)

        # print processed tabular data (uncomment to visualize data structure)
        # for resource in package.resources:
        #     if resource.descriptor['datahub']['type'] == 'derived/csv':
        #         print(resource.read())

        save_package_to_csv(package, data_name)


if __name__ == '__main__':
    main()
