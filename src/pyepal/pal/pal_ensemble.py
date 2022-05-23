import numpy as np


class PALEnsemble:
    def __init__(self, pal_list, reuse_models=False):
        self.pal_list = pal_list

        # we just pick one class where we will update the models
        self.head_pal = pal_list[0]
        self.reuse_models = reuse_models

    @classmethod
    def from_class_and_kwarg_lists(pal_class, **kwargs):

        # Throw error if there are no kwargs
        if not kwargs:
            raise ValueError("No kwargs provided")

        pal_list = []
        iterable_keys = []
        for key, value in kwargs.items():
            if isinstance(value, list, tuple):
                iterable_keys.append(key)

        # the problem is here that we would still need to account for the fact that some arguments by themselves are
        # iterable, but not the others. The coding will be much easier if we just, for every model, accept its kwargs

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
        sample_discarded: bool = False,
        use_coef_var: bool = True,
        replace_mean: bool = True,
        replace_std: bool = True,
    ):
        samples = []
        uncertainties = []
        head_samples, head_uncertainties = self.head_pal.run_one_step(
            batch_size, sample_discarded, use_coef_var, replace_mean, replace_std
        )

        if isinstance(head_samples, int):
            head_samples = [head_samples]
        if isinstance(head_uncertainties, float):
            head_uncertainties = [head_uncertainties]
        uncertainties.extend(head_uncertainties)
        samples.extend(head_samples)

        for pal in self.pal_list[1:]:
            this_samples, this_uncertainties = pal.run_one_step(
                batch_size,
                sample_discarded,
                use_coef_var,
                replace_mean,
                replace_std,
                replacement_models=self.head_pal.models if self.reuse_models else None,
            )

            this_uncertainties = np.array(this_uncertainties)
            this_uncertainties = (
                this_uncertainties - this_uncertainties.mean()
            ) / this_uncertainties.std()
            if isinstance(this_samples, int):
                this_samples = [this_samples]
            if isinstance(this_uncertainties, float):
                this_uncertainties = [this_uncertainties]
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
