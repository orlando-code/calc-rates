import yaml


# function to read in yamls
def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def write_yaml(data, fp="unnamed.yaml"):
    with open(fp, "w") as file:
        yaml.dump(data, file)


def append_to_yaml(data, fp="unnamed.yaml"):
    with open(fp, "a") as file:
        yaml.dump(data, file)
