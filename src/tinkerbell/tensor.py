from re import S
from typing import Self
import numpy as np
from collections import defaultdict

"""
A class representing a tensor in the forward pass of a neural network for forward Autograd.
The result of an operation on tensors is also a tensor.
In this context, a tensor is essentially a multi-dimensional array.
In format x + 𝜖X`:
    x: The value or the result of the tensor operation.
    𝜖: Special symbol with property 𝜖**2 = 0
    X`: The derivative or gradient information associated with the tensor.
"""
class ForwardTensor:
    def __init__(self,primal, requires_grad=False,tangent=None):
        self.primal = np.array(primal)
        self.requires_grad = requires_grad
        if requires_grad and tangent is None:
            self.tangent = np.ones_like(self.primal)
        elif tangent is None:
            self.tangent = np.zeros_like(self.primal)
        else:
            self.tangent = np.array(tangent)
        
    def __add__(self, other):
        if isinstance(other, ForwardTensor):
            primal = self.primal + other.primal
            tangent = self.tangent + other.tangent
            requires_grad = self.requires_grad or other.requires_grad
            return ForwardTensor(primal, requires_grad, tangent)
        else:
            primal = self.primal + other
            return ForwardTensor(primal, self.requires_grad, self.tangent)
    
    def __mul__(self, other):
        if isinstance(other, ForwardTensor):
            primal = self.primal * other.primal
            tangent = self.primal * other.tangent + self.tangent * other.primal
            requires_grad = self.requires_grad or other.requires_grad
            return ForwardTensor(primal, requires_grad, tangent)
        else:
            primal = self.primal * other
            tangent = self.tangent * other
            return ForwardTensor(primal, self.requires_grad, tangent)
    
    def __sub__(self, other):
        if isinstance(other, ForwardTensor):
            primal = self.primal - other.primal
            tangent = self.tangent - other.tangent
            requires_grad = self.requires_grad or other.requires_grad
            return ForwardTensor(primal, requires_grad, tangent)
        else:
            primal = self.primal - other
            return ForwardTensor(primal, self.requires_grad, self.tangent)

    def __truediv__(self, other):
        if isinstance(other, ForwardTensor):
            primal = self.primal / other.primal
            tangent = (self.tangent * other.primal - self.primal * other.tangent) / (other.primal ** 2)
            requires_grad = self.requires_grad or other.requires_grad
            return ForwardTensor(primal, requires_grad, tangent)
        else:
            primal = self.primal / other
            tangent = self.tangent / other
            return ForwardTensor(primal, self.requires_grad, tangent)

    def __repr__(self):
        return f"ForwardTensor(primal={self.primal}, tangent={self.tangent})"


    # def sin(self):
    #     primal = np.sin(self.primal)
    #     tangent = self.tangent * np.cos(self.primal)
    #     return ForwardTensor(primal, self.requires_grad, tangent)


    # def cos(self):
    #     primal = np.cos(self.primal)
    #     tangent = -self.tangent * np.sin(self.primal)
    #     return ForwardTensor(primal, self.requires_grad, tangent)


# ------------------------------- MAIN TENSOR CLASS FOR REVERSE AUTOGRAD ------------------------------- #

class Tensor:
    def __init__(self, data, parents = (), requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(parents)
        self.shape = self.data.shape
        self.ndim = self.data.ndim

    @classmethod
    def handle_broadcasting(cls, grad, target_shape):
        # Sum out leading dimensions if grad has more dimensions
        while grad.ndim > len(target_shape):
            grad = np.sum(grad, axis=0, keepdims=False)
        # Sum out dimensions that were broadcasted (size 1)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = np.sum(grad,axis=i, keepdims=True)
            elif grad.shape[i] != dim:
                raise ValueError("Incompatible shapes for broadcasting during backpropagation.")
        return grad


    def __add__(self, other):
        if isinstance(other, Tensor):
            data = self.data + other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad, self.shape)
                other.grad += Tensor.handle_broadcasting(out.grad, other.shape)
            
        else:
            data = self.data + other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad, self.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            data = self.data * other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad * other.data, self.shape)
                other.grad += Tensor.handle_broadcasting(out.grad * self.data, other.shape)
            
        else:
            data = self.data * other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad * other, self.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        if isinstance(other, Tensor):
            data = self.data - other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad, self.shape)
                other.grad -= Tensor.handle_broadcasting(out.grad, other.shape)
            
        else:
            data = self.data - other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad, self.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            data = self.data / other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad * (1 / other.data), self.shape)
                other.grad -= Tensor.handle_broadcasting(out.grad * (self.data / (other.data ** 2)), other.shape)
            
        else:
            data = self.data / other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += Tensor.handle_broadcasting(out.grad * (1 / other), self.shape)

        out._backward = _backward
        return out

    def __pow__(self, power:int|float):
        data = self.data ** power
        out = Tensor(data, (self,))

        def _backward():
            self.grad += Tensor.handle_broadcasting(out.grad * (power * self.data ** (power - 1)), self.shape)

        out._backward = _backward
        return out

    @classmethod
    def matmul(cls,a:np.ndarray|Self, b:np.ndarray|Self) -> Self:
        if isinstance(a,Tensor) and isinstance(b, Tensor):
            data = a.data @ b.data
            out = cls(data, (a, b))

            def _backward():
                # Use swapaxes for proper batch matmul transpose backprop
                a_grad = out.grad @ (b.data.swapaxes(-1, -2) if b.data.ndim >= 2 else b.data)
                b_grad = (a.data.swapaxes(-1, -2) if a.data.ndim >= 2 else a.data) @ out.grad
                a.grad += cls.handle_broadcasting(a_grad, a.shape)
                b.grad += cls.handle_broadcasting(b_grad, b.shape)

        elif isinstance(a, Tensor):
            data = a.data @ b
            out = cls(data, (a,))

            def _backward():
                a_grad = out.grad @ (b.swapaxes(-1, -2) if b.ndim >= 2 else b)
                a.grad += cls.handle_broadcasting(a_grad, a.shape)

        elif isinstance(b, Tensor):
            data = a @ b.data
            out = cls(data, (b,))

            def _backward():
                b_grad = (a.swapaxes(-1, -2) if a.ndim >= 2 else a) @ out.grad
                b.grad += cls.handle_broadcasting(b_grad, b.shape)

        out._backward = _backward
        return out

    @classmethod 
    def exp(cls, tensor:Self) -> Self:
        data = np.exp(tensor.data)
        out = cls(data, (tensor,))

        def _backward():
            tensor.grad += out.grad * out.data

        out._backward = _backward
        return out

    @classmethod
    def randn(cls, shape:tuple[int,...], requires_grad=False) -> Self:
        if shape is None:
            raise ValueError("Shape must be provided for randn tensor.")
        data = np.random.randn(*shape)
        return cls(data, (), requires_grad)
    
    @classmethod
    def zeros(cls, shape:tuple[int,...], requires_grad=False) -> Self:
        if shape is None:
            raise ValueError("Shape must be provided for zeros tensor.")
        data = np.zeros(shape)
        return cls(data, (), requires_grad)
    
    @classmethod
    def ones(cls, shape:tuple[int,...], requires_grad=False) -> Self:
        if shape is None:
            raise ValueError("Shape must be provided for ones tensor.")
        data = np.ones(shape)
        return cls(data, (), requires_grad)

    @classmethod
    def mean(cls, tensor:Self) -> Self:
        data = np.mean(tensor.data)
        out = cls(data, (tensor,))

        def _backward():
            tensor.grad += out.grad * np.ones_like(tensor.data) / tensor.data.size

        out._backward = _backward
        return out
    

    # NOTE:
    # Potential efficiency improvement with iterative topo sort 

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    # ----------------------------- TODO: Advanced Indexing and Slicing ----------------------------- #
    """ TODO: Implement __getitem__ for advanced indexing and slicing """
    def __getitem__(self, idx):
        data = self.data[idx]
        out = Tensor(data, (self,))
        return out
            

    def __repr__(self):
        return f"Tensor(data={self.data}, size={self.data.shape})"


class Parameter(Tensor):
    def __init__(self, data):
        if isinstance(data, Tensor):
            self.__dict__.update(data.__dict__)
            self.requires_grad = True
        else:
            super().__init__(data,requires_grad=True)



if __name__ == "__main__":
    a = ForwardTensor(2.0, requires_grad=True)
    b = ForwardTensor(3.0, requires_grad=False)
    c = ForwardTensor(7.0, requires_grad=False)

    d = a * b + c*(a + b)
    print("Forward Tensor d:", d)

    a = Tensor(2.0, parents=(), requires_grad=True)
    b = Tensor(3.0, parents=(), requires_grad=True)
    c = Tensor(7.0, requires_grad=True)
    d = a * b + c * (a + b)
    d.backward()
    print("Reverse Tensor a:", a, "with grad")

    a = Tensor(np.random.randn(5,3), requires_grad=True)
    b = Tensor(np.random.randn(3,4), requires_grad=True)
    c = Tensor.matmul(a,b)
    c.backward()
    print("Tensor MatMul c:", c, "with grad", c.grad)
    # Expected output:
    # c.primal = 2.0 * 3.0 + 2.0 / 3.0 - 4.0 = 6.666666666666667
    # c.tangent = (b.primal * a.tangent + a.primal * b.tangent) + (a.tangent / b.primal - a.primal * b.tangent / (b.primal ** 2))
    #            = (3.0 * 1.0 + 2.0 * 1.0) + (1.0 / 3.0 - 2.0 * 1.0 / (3.0 ** 2))
    #            = 5.0 + (0.3333333333333333 - 0.2222222222222222)
    #            = 5.111111111111111

