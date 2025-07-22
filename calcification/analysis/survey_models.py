from calcification.utils import file_ops

file_ops.load_yaml("model_specs.yaml")


def run_model_survey(model_specs):
    for spec in model_specs:
        df = load_data(spec["data_path"])
        model = fit_model(df, spec["formula"], spec["group_col"])
        summary = summarize_model(model)
        save_summary(summary, spec["name"])
        fig = plot_regression(df, model, ...)
        fig.savefig(f"figures/model_plots/{spec['name']}.png")
