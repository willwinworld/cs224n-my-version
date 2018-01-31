#! python2
# -*- coding: utf-8 -*-
import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        row_max = np.max(x, axis=1).reshape(x.shape[0], 1)  # 每一行的最大值
        x = x - row_max  # 将原来的数值较大的矩阵减去每一行的最大值后变成新的矩阵
        exp_x = np.exp(x)  # 对新的矩阵进行指数运算,又变换了一次矩阵
        row_exp_sum = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)  # 横轴方向将指数计算过后的值相加
        x = exp_x / row_exp_sum  # 对减去每行最大值指数计算后的值去除以每一行指数值的和
    else:
        # Vector
        max_vector = np.max(x)  # 选取向量当中最大的元素
        x -= max_vector  # 对向量中的每个元素都减去最大的元素, 向量已经更新了，e.g: [1,2,3] => [-2, -1, 0]
        exp_vector = np.exp(x)  # 通过广播性质进行指数计算传播,并得到新的指数计算后的向量
        x = exp_vector / np.sum(exp_vector)  # 对向量中的每个元素进行除法计算

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    # test_softmax()
