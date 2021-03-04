# from multiprocessing import Process, Queue
#
# def f(q):
#     while not q.full():
#         q.put([42, None, 'hello'])
#
# if __name__ == '__main__':
#     q = Queue(10)
#     p = Process(target=f, args=(q,))
#     p.start()
#     print (q.get())    # prints "[42, None, 'hello']"
#     print (q.get())    # prints "[42, None, 'hello']"
#
#     p.join()
