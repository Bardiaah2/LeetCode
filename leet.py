from typing import *

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self) -> str:  # not defined in LeetCode
        root = self
        result = []
        queue = []
        next_level = [root]

        while next_level:
            queue = next_level
            next_level = []

            for root in queue:
                if root is None:
                    continue
                next_level.append(root.left)
                next_level.append(root.right)
            result.append([i.val for i in queue if i])
            if not result[-1]: result.pop()

        return str(result)

    def __repr__(self) -> str:  # not defined in LeetCode
        return 'TreeNode'


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self) -> str:  # not defined in LeetCode
        temp = self
        result = []
        while temp:
            result.append(temp.val)
            temp = temp.next
        return result.__str__()

    def __repr__(self) -> str:  # not defined in LeetCode
        return 'ListNode'


class Solution:
    def inorderTraversal(self, root: TreeNode):  # 94
        if root.left is None:
            return [root.val] + self.inorderTraversal(root.right) if root.right else [root.val]
        return self.inorderTraversal(root.left) + [root.val]


    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:  # 2
        add = False
        if l1.val + l2.val >= 10:
            add = True
        if l1.next is None and l2.next is None:
            if add:
                return ListNode(l1.val+l2.val % 10, next=ListNode(1))
            return ListNode(l1.val+l2.val % 10)
        else:
            next_l1 = l1.next or ListNode(0)
            next_l2 = l2.next or ListNode(0)
            if add:
                next_l1.val += 1
                return ListNode(l1.val+l2.val % 10, next=self.addTwoNumbers(next_l1, next_l2))
            return ListNode(l1.val+l2.val % 10, next=self.addTwoNumbers(next_l1, next_l2))


    def reverseList(self, head: ListNode) -> ListNode:  # 206
        if head is None:
            return None
        end = head
        while end.next is not None:
            per = end
            end = end.next
        per.next = None
        end.next = self.reverseList(head.next)
        return end


    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:  # 19
        _head = ListNode(next=head)
        head0 = _head
        headn = _head
        for _ in range(n):
            headn = headn.next
        while headn.next is not None:
            headn = headn.next
            head0 = head0.next
        head0.next = head0.next.next
        return _head.next


    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:  # 160
        a, b = headA, headB

        while a is not b:
            if a is None:
                a = headB
            else:
                a = a.next
            if b is None:
                b = headA
            else:
                b = b.next

        return a


    def swapPairs(self, head: ListNode, per_head: ListNode | None = None) -> ListNode:  # 24
        if per_head is None:
            per_head = ListNode(next=head)
        # swap the first two nodes:
        #   1. check that those two are not None:
        if head is None or head.next is None:
            return head
        #   2. swap:
        per_head.next = head.next
        head.next = head.next.next
        per_head.next.next = head
        self.swapPairs(head.next, head)
        return per_head.next


    def hasCycle(self, head: ListNode) -> bool:  # 141
        walker, runner = head, head
        while runner and runner.next:
            runner = runner.next.next
            walker = walker.next
            if runner is walker:
                return True
        return False


    def detectCycle(self, head: ListNode) -> ListNode:  # 142
        head1, head_fast, head_slow = head, head, head
        if head is None:
            return None
        while head_fast.next is not None:
            head_fast = head_fast.next.next
            head_slow = head_slow.next
            if head_fast is head_slow:
                head1 = head1.next
            if head_fast is None:
                return None
            if head_fast is head1 or head_slow is head1:
                return head1

        return None


    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:  # 21
        if not list1 or not list2:
            if lis := list1 or list2:
                return ListNode(lis.val, self.mergeTwoLists(lis.next, None))
            else:
                return None
        if list1.val < list2.val:
            list2, list1 = list1, list2

        return ListNode(list2.val, self.mergeTwoLists(list2.next, list1))


    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:  # 102
        result = []
        queue = []
        next_level = [root]

        while next_level:
            queue = next_level
            next_level = []

            for root in queue:
                if root is None:
                    continue
                next_level.append(root.left)
                next_level.append(root.right)
            result.append([i.val for i in queue if i])
            if not result[-1]: result.pop()

        return result


    def isSymmetric(self, root: Optional[TreeNode]) -> bool:  # 101
        def wrapper(root_right: TreeNode, root_left: TreeNode) -> bool:
            if not root_left or not root_right:
                return isinstance(root_left, root_right.__class__)
            return root_right.val == root_left.val and wrapper(root_right.left, root_left.right) and wrapper(root_left.left, root_right.right)
        return wrapper(root.right, root.left)


    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:  # 112
        if root is None:
            return False
        if root.right is root.left:
            return targetSum == root.val
        return self.hasPathSum(root.right, targetSum - root.val) or self.hasPathSum(root.left, targetSum - root.val)


    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:  # 328
        if head is None or head.next is None or head.next.next is None:
            return head
        oddHead = head
        end = head
        while end.next:
            end = end.next
        oddEnd = end
        while head is not oddEnd:
            end.next = head.next
            head.next = head.next.next
            end = end.next
            end.next = None
            if end is oddEnd:
                return oddHead
            head = head.next

        return oddHead


    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:  # 145
        if root is None:
            return []

        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]


    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:  # 94
        if root is None:
            return []

        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)


    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:  # 106
        # inorder is first comes left then root then right
        # postorder is first comes left then right then root
        # both methods are DFS
        if not inorder:
            return None

        temp_root = postorder.pop()
        # spliting the list
        index = inorder.index(temp_root)
        right_inorder, left_inorder = inorder[index+1:], inorder[:index]

        if len(right_inorder) == 1:
            root = TreeNode(temp_root, right=TreeNode(right_inorder[0]))
        else:
            root = TreeNode(temp_root, right=self.buildTree(right_inorder, postorder))

        if len(left_inorder) == 1:
            root.left = TreeNode(left_inorder[0])
        else:
            root.left = self.buildTree(left_inorder, postorder)

        return root


    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:  # 108
        pass


    def countSmaller(self, nums: List[int]) -> List[int]:  # 315
        pass
