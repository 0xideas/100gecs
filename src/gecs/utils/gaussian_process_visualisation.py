import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.utils.extmath import cartesian


def visualise_1D_gaussian_process(gaussian):
    x = np.atleast_2d(np.linspace(0, 10, 100)).T
    y_pred, sigma = gaussian.predict(x, return_std=True)
    plt.figure(figsize=(15, 15))
    plt.plot(x, y_pred, "b-", label="Prediction")

    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=0.5,
        fc="b",
        ec="None",
        label="95% confidence interval",
    )
    plt.show()


def visualise_2D_gaussian_process(
    gaussian,
    X_range=np.arange(0, 10, 0.1),
    Y_range=np.arange(0, 10, 0.1),
    Z_range=np.arange(-3, 3, 1),
    plot_bounds=True,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 15))

    # Make data.
    X = np.arange(
        np.min(X_range), np.max(X_range), (np.max(X_range) - np.min(X_range)) / 100
    )
    Y = np.arange(
        np.min(Y_range), np.max(Y_range), (np.max(Y_range) - np.min(Y_range)) / 100
    )
    Z = np.arange(
        np.min(Z_range), np.max(Z_range), (np.max(Z_range) - np.min(Z_range)) / 100
    )
    R = cartesian(np.array([X, Y]))
    X, Y = np.meshgrid(X, Y)
    Z, sigma = gaussian.predict(R, return_std=True)
    Z = Z.reshape((100, 100))
    sigma = sigma.reshape((100, 100))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if plot_bounds:
        surf2 = ax.plot_surface(
            X, Y, Z + 1.9600 * sigma, alpha=0.2, linewidth=0, antialiased=False
        )
        surf3 = ax.plot_surface(
            X, Y, Z - 1.9600 * sigma, alpha=0.2, linewidth=0, antialiased=False
        )
    # Customize the z axis.
    ax.set_zlim(np.min(Z_range), np.max(Z_range))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
