from leet import *
import unittest

null = None

class MakeClass:
    '''A class that would make a class of the given class with the given value'''
    def __init__(self, cls, value) -> None:
        self.cls = cls
        self.value = value


    def __TreeNode(self) -> TreeNode:
        if not self.value:
            return None
        value = self.value
        # make a root
        root = TreeNode(value.pop(0))
        queue = [root]

        while value:
            temp_root = queue.pop()

            # add the next two to right and left
            left_val, right_val = value.pop(0), value.pop(0)
            if temp_root is None:
                queue = [null, null] + queue
                continue
            if left_val: temp_root.left = TreeNode(left_val)
            if right_val: temp_root.right = TreeNode(right_val)

            # add the right and left to queue
            queue = [temp_root.right, temp_root.left] + queue

        return root


    def __ListNode(self) -> ListNode:
        value = self.value
        # edge case
        if not value:
            return None
        # reverse traversal adding to the linked list
        per = ListNode(value.pop())
        while value:
            result = ListNode(value.pop(), per)
            per = result

        return result


    def get(self) -> Any:
        return self.__getattribute__('_MakeClass__' + self.cls().__repr__())()


class Test(unittest.TestCase):
    '''This class is for making test cases using methods that usually start
    with the word, test, and the name of the function its testing'''
    def test_levelOrder(self):
        root = [3,9,20,null,null,15,7]
        root = MakeClass(TreeNode, root).get()
        self.assertEqual(Solution().levelOrder(root), [[3],[9,20],[15,7]])


    def test_isSymmetric(self):
        # test case 1
        root = [1,2,2,3,4,4,3]
        root = MakeClass(TreeNode, root).get()
        self.assertEqual(Solution().isSymmetric(root), True)
        # test case 2
        root = [1,2,2,null,3,null,3]
        root = MakeClass(TreeNode, root).get()
        self.assertEqual(Solution().isSymmetric(root), False)


    def test_hasPathSum(self):
        # test case 1
        root = [5,4,8,11,null,13,4,7,2,null,null,null,1]
        root = MakeClass(TreeNode, root).get()
        self.assertEqual(Solution().hasPathSum(root, 22), True)
        # test case 2
        root = [1,2,3]
        root = MakeClass(TreeNode, root).get()
        self.assertEqual(Solution().hasPathSum(root, 5), False)
        # test case 3
        root = []
        root = MakeClass(TreeNode, root).get()
        self.assertEqual(Solution().hasPathSum(root, 0), False)


    def test_oddEvenList(self):
        # test case 1
        head = [1,2,3,4,5]
        head = MakeClass(ListNode, head).get()
        # print(Solution().oddEvenList(head))
        self.assertEqual(str(Solution().oddEvenList(head)), str(MakeClass(ListNode, [1,3,5,2,4]).get()))
        # test case 2
        head = [2,1,3,5,6,4,7]
        head = MakeClass(ListNode, head).get()
        self.assertEqual(str(Solution().oddEvenList(head)), str(MakeClass(ListNode, [2,3,6,7,1,5,4]).get()))
        # test case 3
        head = [1,2,3,4,5,6,7,8]
        head = MakeClass(ListNode, head).get()
        self.assertEqual(str(Solution().oddEvenList(head)), str(MakeClass(ListNode, [1,3,5,7,2,4,6,8]).get()))


if __name__ == '__main__':
    unittest.main()
