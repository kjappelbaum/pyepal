import numpy as np


class PALEnsemble:
    def __init__(self, pal_list):
        self.pal_list = pal_list

        # we just pick one class where we will update the models
        self.head_pal = pal_list[0]

    @classmethod
    def from_class_and_kwarg_lists(pal_class, **kwargs):
        pal_list = []
        iterable_keys = []
        for key, value in kwargs.items():
            if isinstance(value, list, tuple):
                iterable_keys.append(key)

        if len(iterable_keys) == 0:
            raise ValueError(
                "No iterable keys found in kwargs. If you do not provide iterable keys, please use a single PAL instance."
            )

        num_values = len(kwargs[iterable_keys[0]])

        for key in iterable_keys:
            if len(kwargs[key]) != num_values:
                raise ValueError(
                    "All iterable keys must have the same length. Please check the length of your iterable keys."
                )

        for i in range(num_values):
            this_kwargs = {}
            for key, value in kwargs.items():
                if key in iterable_keys:
                    this_kwargs[key] = value[i]
                else:
                    this_kwargs[key] = value
            pal_list.append(pal_class(**this_kwargs))
        return PALEnsemble(pal_list)

    def run_one_step(
        self,
        batch_size: int = 1,
        pooling_method: str = "fro",
        sample_discarded: bool = False,
        use_coef_var: bool = True,
        replace_mean: bool = True,
        replace_std: bool = True,
    ):
        samples = []
        uncertainties = []
        head_samples, head_uncertainties = self.head_pal.run_one_step(
            batch_size, pooling_method, sample_discarded, use_coef_var, replace_mean, replace_std
        )
        samples.extend(head_samples)
        uncertainties.extend(head_uncertainties)

        samples.extend(head_samples)

        for pal in self.pal_list[1:]:
            this_samples, this_uncertainties = pal.run_one_step(
                batch_size,
                pooling_method,
                sample_discarded,
                use_coef_var,
                replace_mean,
                replace_std,
                replace_models=self.head_pal.models,
            )
            samples.extend(this_samples)
            uncertainties.extend(this_uncertainties)

        uncertainties_sorted, indices_sorted = zip(*sorted(zip(uncertainties, samples)))
        uncertainties_sorted = np.array(uncertainties_sorted)
        indices_sorted = np.array(indices_sorted)
        _, original_sorted_indices = np.unique(indices_sorted, return_index=True)
        indices_selected = indices_sorted[original_sorted_indices]
        return indices_selected[-batch_size:], uncertainties_sorted[-batch_size:]

    def augment_design_space(  # pylint: disable=invalid-name
        self, X_design: np.ndarray, classify: bool = False, clean_classify: bool = True
    ) -> None:
        for pal in self.pal_list:
            pal.augment_design_space(X_design, classify, clean_classify)

    def update_train_set(
        self,
        indices: np.ndarray,
        measurements: np.ndarray,
        measurement_uncertainty: np.ndarray = None,
    ) -> None:
        for pal in self.pal_list:
            pal.update_train_set(indices, measurements, measurement_uncertainty)
