"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    products = matrices @ vectors
    return products.sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(matrix.dtype if matrix.dtype.kind in "iu" else int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [list(np.unique(matrix[:, j])) for j in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    row_vars = matrix.var(axis=1)
    col_vars = matrix.var(axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=col_means,
        row_variances=row_vars,
        column_variances=col_vars,
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i = np.arange(rows)[:, None]
    j = np.arange(columns)[None, :]
    mask = (i + j) % 2 == 0
    return np.where(mask, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    cy, cx = image_height // 2, image_width // 2

    top = cy - height // 2
    bottom = top + height
    left = cx - width // 2
    right = left + width

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, image_height)
    right = min(right, image_width)

    image[top:bottom, left:right] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    cy, cx = image_height / 2.0, image_width / 2.0
    y, x = np.ogrid[:image_height, :image_width]

    mask = ((x - cx) ** 2) / (semi_axis_x ** 2) + ((y - cy) ** 2) / (semi_axis_y ** 2) <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)

    mean = float(values.mean())
    variance = float(values.var())
    std = float(values.std())

    # строгие локальные максимумы/минимумы (без крайних элементов)
    local_max = np.where(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    )[0] + 1
    local_min = np.where(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    )[0] + 1

    # скользящее среднее с окном window
    if window > 0 and window <= len(values):
        kernel = np.ones(window) / window
        moving_avg = np.convolve(values, kernel, mode="valid")
    else:
        moving_avg = np.array([], dtype=float)

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_max.tolist(),
        local_minima_indices=local_min.tolist(),
        moving_average=moving_avg,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size > 0 else 0

    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result

#1
mats = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
vecs = np.array([[[1], [1]], [[1], [1]]])
res = sum_prod(MatrixVectorBatchInput(mats, vecs))
print(res.ravel().tolist(), "== [3, 3]")

mats1 = np.array([[[1, 2], [3, 4]]])
vecs1 = np.array([[[1], [1]]])
print(sum_prod(MatrixVectorBatchInput(mats1, vecs1)).ravel().tolist(), "== [3, 7]")

#2
m = np.array([[1, 5], [7, 2]])
b = binarize(BinarizeInput(m, 4))
print(b.tolist(), "== [[0, 1], [1, 0]]")
print("исходная не изменилась:", m.tolist(), "== [[1, 5], [7, 2]]")
print(binarize(BinarizeInput(np.array([[0, 0], [0, 0]]), 0)).tolist(), "== [[0, 0], [0, 0]]")

#3
m = np.array([[3, 1, 3, 2], [5, 5, 5, 5]])
print(unique_rows(MatrixInput(m)), "== [[1, 2, 3], [5]]")
print(unique_columns(MatrixInput(m)), "== [[3, 5], [1, 5], [3, 5], [2, 5]]")
m2 = np.array([[1.5, 1.5, 2.5]])
print(unique_rows(MatrixInput(m2)), "== [[1.5, 2.5]]")

#4
stat = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=0, std=1, seed=42))
print("shape:", stat.matrix.shape, "== (3, 4)")
print("row_means:", stat.row_means.shape, "== (3,)")
print("column_means:", stat.column_means.shape, "== (4,)")
stat2 = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=0, std=1, seed=42))
print("воспроизводимость:", np.allclose(stat.matrix, stat2.matrix), "== True")

#5
c = chess(ChessInput(rows=3, columns=4, first=0, second=1))
print(c.tolist())
print("== [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]")
c2 = chess(ChessInput(rows=1, columns=1, first=5, second=9))
print(c2.tolist(), "== [[5]]")
c3 = chess(ChessInput(rows=2, columns=2, first=7, second=3))
print(c3.tolist(), "== [[7, 3], [3, 7]]")

#6
img = draw_rectangle(RectangleInput(
    width=2, height=2, image_height=5, image_width=5,
    shape_color=(255, 0, 0), background_color=(0, 0, 0),
))
print("shape:", img.shape, "== (5, 5, 3)")
print("центр:", img[2, 2].tolist(), "== [255, 0, 0]")
print("угол:", img[0, 0].tolist(), "== [0, 0, 0]")

img = draw_ellipse(EllipseInput(
    semi_axis_x=2, semi_axis_y=2, image_height=5, image_width=5,
    shape_color=(0, 255, 0), background_color=(0, 0, 0),
))
print("shape:", img.shape, "== (5, 5, 3)")
print("центр:", img[2, 2].tolist(), "== [0, 255, 0]")
print("угол:", img[0, 0].tolist(), "== [0, 0, 0]")

#7
ts = analyze_time_series(TimeSeriesInput(values=[1, 3, 2, 5, 4, 6, 0], window=3))
print("mean:", ts.mean, "== 3.0")
print("variance:", round(ts.variance, 4))
print("std:", round(ts.std, 4))
print("local_maxima_indices:", ts.local_maxima_indices, "== [1, 3, 5]")
print("local_minima_indices:", ts.local_minima_indices, "== [2, 4]")
print("moving_average:", np.round(ts.moving_average, 4).tolist(),"== [2.0, 3.3333, 3.6667, 5.0, 3.3333]")

#8
oh = one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=None))
print(oh.tolist())
print("== [[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]]")
oh2 = one_hot(OneHotInput(labels=[0, 1], class_count=5))
print("shape:", oh2.shape, "== (2, 5)")
print(oh2.tolist(), "== [[1,0,0,0,0],[0,1,0,0,0]]")