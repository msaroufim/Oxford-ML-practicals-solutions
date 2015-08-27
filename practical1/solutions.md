# Introduction to Lua and Torch

## Lua Basics

[Lua in 15min](http://tylerneylon.com/a/learn-lua/)

* Why is the local keyword important: default variable scope is not local in Lua
* What is the difference between ```a.f()``` and ```a:f()```: ```a:f()``` is the same as ```a.f(self,...)```
* What does ```require``` do: standard way of including modules. It's return value are cached so that's its run at most once
* What's a list in Lua? What's a dictionary? How do you iterate over them: The only data structure in Lua is a table
    - Dictionary: ```t = {key1 = 'value1', key2 = false}```
    - List: ```v = {'value1', 'value2', 1.21, 'gigawatts'}```
    - Iteration: ```for i = 1, #v do print(v[i]) end```

## Torch Basics

Want to do most operations in place to save memory

```lua
local t = torch.Tensor(10,10)
local t2 = torch.Tensor(10,10)
t:add(t2)
```

### [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md)

```lua
--create 4D-tensor of size 4x5x6x2
--number of potential dimensions is unlimited
z = torch.Tensor(4,5,6,2)
```

Tensor is derived from ```Storage``` which allows ```Lua``` to access memory of a ```C``` pointer. ```Storage``` does not store memory in a contiguous fashion it uses a ```stride(i)``` to determine how to get from the ```(i-1)``` th element to the ```i```th element.

There are many convenient ways to create a tensor, one can pass a table as an argument

```
> torch.Tensor({{1,2,3,4}, {5,6,7,8}})
 1  2  3  4
 5  6  7  8
[torch.DoubleTensor of dimension 2x4]
```

Some useful operations include:

* transposing tensors which can be generalized to permuting the tensors: ```y = x:t() ```
* Apply a function to all elements of a tensor: ```z:apply(function(x) i = i + 1 return i end)```


### [Math](https://github.com/torch/torch7/blob/master/doc/maths.md)

* 2D convolution as simple as: ```res1 = torch.conv2(x,k)```
* concatenate two tensors along a dimension: ```torch.cat(torch.ones(3),torch.zeros(2))```

## Handin

```
a = torch.Tensor{{1,2,3},{4,5,6},{7,8,9}}
> c = a:select(1,2)
> print(c)
 4
 5
 6
[torch.DoubleTensor of dimension 3]
```
1. 


