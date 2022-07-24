from pathlib import Path
import h5netcdf
import rasterio
from affine import Affine
import datetime
import numpy as np
from scipy.stats import expon
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import click
from pynoise.noisemodule import Perlin, ScaleBias
from pynoise.noiseutil import noise_map_plane


def noisy_hillslope(nrows, ncols, bottom, top, n1div=50, n2div=25, n3div=100,
                    n2scale=2, n3scale=1, zroot=1, zpower=3):
    """Generates a hillslope elevation grid with a noisy slope

    Args
    ----------
    nrows : float
        number of rows

    ncols : float
        number of columns

    bottom : float
        elevation at lowest point of hillslope

    top : float
        elevation at highest point of hillslope

    Returns
    ----------
    ZZ : float
        elevation grid
    """
    p = Perlin(frequency=.15, persistence=0.5, octaves=4, seed=21)
    p1 = ScaleBias(source0=p, scale=0.33, bias=0.5)
    noisemap = noise_map_plane(nrows, ncols, 6, 10, 1, 5, p1).reshape((nrows, ncols))
    z_diff = top - bottom
    ZZ = bottom + z_diff * noisemap

    return ZZ


def straight_hillslope(nrows, ncols, bottom, top):
    """Generates a hillslope elevation grid with a straight slope

    Args
    ----------
    nrows : float
        number of rows

    ncols : float
        number of columns

    bottom : float
        elevation at lowest point of hillslope

    top : float
        elevation at highest point of hillslope

    Returns
    ----------
    ZZ : float
        elevation grid
    """
    zs = np.linspace(0, 1, nrows)[::-1]
    ZS = np.zeros((nrows, ncols))
    ZS[:, :] = zs[:, np.newaxis]
    z_diff = top - bottom
    ZZ = bottom + z_diff * ZS

    return ZZ


def concave_hillslope(nrows, ncols, bottom, top):
    """Generates a hillslope elevation grid with a concave slope

    Args
    ----------
    nrows : float
        number of rows

    ncols : float
        number of columns

    bottom : float
        elevation at lowest point of hillslope

    top : float
        elevation at highest point of hillslope

    Returns
    ----------
    ZZ : float
        elevation grid
    """
    x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), nrows)
    zs = expon.pdf(x, scale=1.5)
    ZS = np.zeros((nrows, ncols))
    ZS[:, :] = zs[:, np.newaxis]
    z_diff = top - bottom
    ZZ = bottom + z_diff * ZS

    return ZZ


def convex_hillslope(nrows, ncols, bottom, top):
    """Generates a hillslope elevation grid with a convex slope

    Args
    ----------
    nrows : float
        number of rows

    ncols : float
        number of columns

    bottom : float
        elevation at lowest point of hillslope

    top : float
        elevation at highest point of hillslope

    Returns
    ----------
    ZZ : float
        elevation grid
    """
    x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), nrows)
    zs = 1 - expon.pdf(x, scale=1.5)
    ZS = np.zeros((nrows, ncols))
    ZS[:, :] = zs[:, np.newaxis]
    z_diff = top - bottom
    ZZ = bottom + z_diff * ZS

    return ZZ


def calculate_slope(ZZ, cell_width):
    """Calculates slope in x-direction and y-direction

    Args
    ----------
    ZZ : np.ndarray
        elevation grid

    cell_width : float
        width of grids (in meter)

    Returns
    ----------
    SSY : np.ndarray
        slope grid in y-direction

    SSX : np.ndarray
        slope grid in x-direction
    """
    SSY = np.abs(np.diff(ZZ[:-1, 1:-1], axis=0)/cell_width)
    SSX = np.abs(np.diff(ZZ[1:-1, :-1], axis=1)/cell_width)

    return SSX, SSY


def plot_elevation_grid(ZZ, cell_width):
    """Plots elevation grid

    Args
    ----------
    ZZ : np.ndarray
        elevation grid

    cell_width : float
        width of grids (in meter)
    """
    ZZ = ZZ[1:-1, 1:-1]
    nrows = ZZ.shape[0]
    ncols = ZZ.shape[1]
    X = np.arange(-int(nrows/2)*cell_width, int(nrows/2)*cell_width, cell_width)
    Y = np.arange(-int(ncols/2)*cell_width, int(ncols/2)*cell_width, cell_width)
    YY, XX = np.meshgrid(Y, X)

    zmin = int(np.floor(np.min(ZZ)))
    zmax = int(np.ceil(np.max(ZZ)))

    norm = matplotlib.colors.Normalize(vmin=zmin, vmax=zmax)
    scamap = plt.cm.ScalarMappable(cmap='BrBG', norm=norm)
    fcolors = scamap.to_rgba(ZZ)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, cmap=cm.BrBG,
                    vmin=zmin, vmax=zmax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlim(-int(nrows/2)*cell_width, int(nrows/2)*cell_width)
    ax.set_ylim(-int(ncols/2)*cell_width, int(ncols/2)*cell_width)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'[m a.s.l.]')
    fig.colorbar(scamap, shrink=0.4, label=r'slope [m $m^{-1}$]')

    return fig


def plot_slope_grid(ZZ, SS, cell_width):
    """Plots elevation grid with slope

    Args
    ----------
    ZZ : np.ndarray
        elevation grid

    cell_width : float
        width of grids (in meter)
    """
    ZZ = ZZ[1:-1, 1:-1]
    nrows = ZZ.shape[0]
    ncols = ZZ.shape[1]
    X = np.arange(-int(nrows/2)*cell_width, int(nrows/2)*cell_width, cell_width)
    Y = np.arange(-int(ncols/2)*cell_width, int(ncols/2)*cell_width, cell_width)
    YY, XX = np.meshgrid(Y, X)

    smin = np.ceil(np.min(SS))
    smax = np.floor(np.max(SS))

    norm = matplotlib.colors.Normalize(vmin=smin, vmax=smax)
    scamap = plt.cm.ScalarMappable(cmap='BrBG', norm=norm)
    fcolors = scamap.to_rgba(SS)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, cmap=cm.BrBG,
                    vmin=smin, vmax=smax)
    ax.set_zlim(0, 25)
    ax.set_xlim(-int(ncols/2)*cell_width, int(ncols/2)*cell_width)
    ax.set_ylim(-int(ncols/2)*cell_width, int(ncols/2)*cell_width)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'[m a.s.l.]')
    fig.colorbar(scamap, shrink=0.4, label=r'slope [m $m^{-1}$]')

    return fig


def write_to_netcdf(ZZ, SSY, SSX, cell_width):
    """Write output to Netcdf

    Args
    ----------
    ZZ : np.ndarray
        elevation grid

    SSY : np.ndarray
        slope grid in y-direction

    SSX : np.ndarray
        slope grid in x-direction
    """
    base_path = Path(__file__).parent
    file = base_path / "hillslope.nc"
    with h5netcdf.File(file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Artificial hillslope',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='',
            )
        dict_dim = {'x': SSY.shape[0], 'y': SSY.shape[1], 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable('x', ('x',), float)
        v.attrs['long_name'] = 'Distance in x-direction'
        v.attrs['units'] = 'm'
        v[:] = np.arange(dict_dim["x"]) * cell_width
        v = f.create_variable('y', ('y',), float)
        v.attrs['long_name'] = 'Distance in y-direction'
        v.attrs['units'] = 'm'
        v[:] = np.arange(dict_dim["y"]) * cell_width
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = cell_width
        v = f.create_variable('ELEV', ('x', 'y'), float)
        v.attrs['long_name'] = 'Elevation'
        v.attrs['units'] = 'm'
        v[:, :] = ZZ[1:-1, 1:-1]
        v = f.create_variable('SLOPE_Y', ('x', 'y'), float)
        v.attrs['long_name'] = 'Slope'
        v.attrs['units'] = 'm/m'
        v[:, :] = SSY
        v = f.create_variable('SLOPE_X', ('x', 'y'), float)
        v.attrs['long_name'] = 'Slope'
        v.attrs['units'] = 'm/m'
        v[:, :] = SSX

def write_to_tiff(ZZ, cell_width):
    """Write output to tiff

    Args
    ----------
    ZZ : np.ndarray
        elevation grid
    """
    nrows, ncols = ZZ.shape
    profile = {
        'driver' : 'GTiff',
        'blockxsize' : 256,
        'blockysize' : 256,
        'count': 1,
        'tiled' : True,
        'crs' : '+proj=longlat +datum=WGS84 +units=m +no_defs',
        'transform' : Affine(cell_width, 0.0, 0.0,
                             0.0, cell_width, 0.0),
        'dtype' : 'float64',
        'nodata' : -9999,
        'height' : nrows,
        'width' : ncols
    }
    base_path = Path(__file__).parent
    file = base_path / "hillslope.tiff"
    with rasterio.open(file, 'w', **profile) as dst:
        dst.write(np.asarray(ZZ), 1)


@click.option("-hs", "--hillslope-shape", type=click.Choice(['straight', 'concave', 'convex', 'noisy']), default='straight')
@click.option("-nr", "--nrows", type=int, default=24)
@click.option("-nc", "--ncols", type=int, default=12)
@click.option("-b", "--bottom", type=float, default=10)
@click.option("-t", "--top", type=float, default=12)
@click.option("-cw", "--cell-width", type=float, default=1)
@click.option("--plot", is_flag=True)
@click.option("--write-output", is_flag=True)
@click.command()
def main(hillslope_shape, nrows, ncols, bottom, top, cell_width, plot, write_output):
    if hillslope_shape == 'straight':
        ZZ = straight_hillslope(nrows, ncols, bottom, top)

    elif hillslope_shape == 'concave':
        ZZ = concave_hillslope(nrows, ncols, bottom, top)

    elif hillslope_shape == 'convex':
        ZZ = convex_hillslope(nrows, ncols, bottom, top)

    elif hillslope_shape == 'noisy':
        ZZ = noisy_hillslope(nrows, ncols, bottom, top)

    SSX, SSY = calculate_slope(ZZ, cell_width)

    if plot:
        plot_elevation_grid(ZZ, cell_width)
        # plot_slope_grid(ZZ, SSY, cell_width)

    if write_output:
        write_to_netcdf(ZZ, SSY, SSX, cell_width)
        write_to_tiff(ZZ, cell_width)
    return


if __name__ == "__main__":
    main()
    # main('straight', 24, 12, 10, 12, 1, True, True)
    # main('concave', 24, 12, 10, 12, 1, True, True)
    # main('noisy', 24, 12, 10, 12, 1, True, True)
