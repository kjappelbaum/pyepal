# -*- coding: utf-8 -*-
# def _ensemble_train_one_finite_width(
#     i: int,
#     models: Sequence[NTModel],
#     design_space: np.ndarray,
#     objectives: np.ndarray,
#     sampled: np.ndarray,
#     opt_init,
#     opt_update,
#     get_params,
#     training_steps,
#     ensemble_size,
# ):
#     model = models[i]
#     loss = jit(lambda params, x, y: 0.5 * np.mean((model.apply_fn(params, x) - y) ** 2))
#     grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

#     x_train = design_space[sampled[:, i]]
#     y_train = objectives[sampled[:, i], i].reshape(-1, 1)

#     def train_network(key):
#         _, params = model.init_fn(key, (-1, x_train.shape[1]))
#         opt_state = opt_init(params)

#         for i in range(training_steps):
#             opt_state = opt_update(i, grad_loss(opt_state, x_train, y_train), opt_state)

#         return get_params(opt_state)

#     ensemble_key = random.split(KEY, ensemble_size)
#     params = vmap(train_network)(ensemble_key)

#     ensemble_func = vmap(model.apply_func, (0, None))(params, test_xs)

#     mean_func = np.reshape(np.mean(ensemble_func, axis=0), (-1,))
#     std_func = np.reshape(np.std(ensemble_func, axis=0), (-1,))
