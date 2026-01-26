import numpy as np

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
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(parents)


    def __add__(self, other):
        if isinstance(other, Tensor):
            data = self.data + other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            
        else:
            data = self.data + other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            data = self.data * other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            
        else:
            data = self.data * other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += other * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        if isinstance(other, Tensor):
            data = self.data - other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += out.grad
                other.grad -= out.grad
            
        else:
            data = self.data - other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            data = self.data / other.data
            out = Tensor(data, (self, other))

            def _backward():
                self.grad += (1 / other.data) * out.grad
                other.grad -= (self.data / (other.data ** 2)) * out.grad
            
        else:
            data = self.data / other
            out = Tensor(data, (self,))
            def _backward():
                self.grad += (1 / other) * out.grad

        out._backward = _backward
        return out

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
            

    def __repr__(self):
        return f"Tensor(data={self.data}, size={self.data.shape})"





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
    # Expected output:
    # c.primal = 2.0 * 3.0 + 2.0 / 3.0 - 4.0 = 6.666666666666667
    # c.tangent = (b.primal * a.tangent + a.primal * b.tangent) + (a.tangent / b.primal - a.primal * b.tangent / (b.primal ** 2))
    #            = (3.0 * 1.0 + 2.0 * 1.0) + (1.0 / 3.0 - 2.0 * 1.0 / (3.0 ** 2))
    #            = 5.0 + (0.3333333333333333 - 0.2222222222222222)
    #            = 5.111111111111111

