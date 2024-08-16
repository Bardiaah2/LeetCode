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


class Node:
    def __init__(self, val: int, next: Optional['Node'], random: Optional['Node']) -> None:
        self.val = val
        self.next = next
        self.random = random

    def __str__(self) -> str:
        temp = self
        result = []
        while temp:
            if temp.random: result.append([temp.val, temp.random.val])
            else: result.append([temp.val, None])
            temp = temp.next
        return result.__str__()

    def __repr__(self) -> str:
        return 'Node'


class Solution:
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


    def mergeTwoLists(self, list1: ListNode | None, list2: ListNode | None) -> ListNode:  # 21
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
        def wrapper(root_right: TreeNode | None, root_left: TreeNode | None) -> bool:
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


    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:  # 144
        if root is None:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)


    def buildTree105(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:  # 106
        # inorder left root right [[left.left left.val left.right] root.val [right.left right.val right.right]]
        # postorder left right root [[left.left left.right left.val] [right.left right.right right.val] root.val]
        if not postorder:
            return None

        root = TreeNode(postorder.pop())
        # split inorder from root.val
        index = inorder.index(root.val)
        left_inorder, right_inorder = inorder[:index], inorder[index+1:]
        # split postorder from right.left
        left_postorder, right_postorder = postorder[:len(left_inorder)], postorder[len(left_inorder):]

        root.right = self.buildTree105(right_inorder, right_postorder)
        root.left = self.buildTree105(left_inorder, left_postorder)
        return root


    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:  # 108
        if not nums:
            return None
        middle_index = (len(nums)-1) // 2
        root = TreeNode(nums.pop(middle_index))
        nums_left, nums_right  = nums[:middle_index], nums[middle_index:]

        root.right = self.sortedArrayToBST(nums_right)
        root.left = self.sortedArrayToBST(nums_left)

        return root


    def isPalindrome(self, head: Optional[ListNode]) -> bool:  # 234
        # find the middle
        end = head
        middle = head
        while end.next and end.next.next:
            end = end.next.next
            middle = middle.next
        # reverse the second half
        end = middle.next
        middle.next = None
        while end:
            per = end
            end = end.next
            per.next = middle
            middle = per

        # compare the two parts
        while middle and head:
            if head.val != middle.val:
                return False
            head = head.next
            middle = middle.next

        return True


    def buildTree106(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:  # 105
        # inorder: [left, val, right]
        # preorder: [val, left, right]
        if not preorder:
            return None
        # make root
        root = TreeNode(preorder.pop(0))
        # split inorder
        index = inorder.index(root.val)
        left_inorder, right_inorder = inorder[:index], inorder[index+1:]
        # split preorder
        left_preorder, right_preorder = preorder[:len(left_inorder)], preorder[len(left_inorder):]

        root.left, root.right = self.buildTree106(left_preorder, left_inorder), self.buildTree106(right_preorder, right_inorder)

        return root


    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:  # 2
        head = end = ListNode()
        while True:
            end.val += l1.val if l1 else 0
            end.val += l2.val if l2 else 0
            if end.val >= 10:
                end.val %= 10
                end.next = ListNode(1)
            l1, l2 = l1.next if l1 else None, l2.next if l2 else None
            if not l1 and not l2:
                return head
            else:
                end.next = end.next if end.next else ListNode(0)
                end = end.next


    def removeElement(self, nums: List[int], val: int) -> int:  # 27
        # two pointer technique
        i, j = 0, len(nums)-1
        if j < 0:
            return 0
        while i < j:
            # if nums[j] == val: j--
            if nums[j] == val:
                j -= 1
            # if nums[i] == val: nums[i] = nums[j]; j-- i++
            elif nums[i] == val:
                nums[i], nums[j] = nums[j], nums[i]
                j -= 1
                i += 1
            # if nums[i] != val: i++ count++
            else:
                i += 1
        # return i + 1 if  the last item is not value of val
        return i + (nums[i] != val)


    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:  # 61
        if not head or not head.next or not k:
            return head
        # have end and per
        end, per = head, head

        end = head
        # seperate end and per by k%n  nodes
        for i in range(k):  # O(n)
            end = end.next
            if end is None:
                end = head
                for _ in range(k%(i+1)):
                    end = end.next
                break

        # moving end and per together till end hits the end
        while end.next:  # O(n)
            end = end.next
            per = per.next

        result = per.next
        if not result:
            return head
        per.next = None
        end.next = head
        return result


    def isBalanced(self, root: Optional[TreeNode]) -> bool:  # 110
        def height(root: Optional[TreeNode]) -> int:
            if not root:
                return 0
            return 1+max(height(root.right), height(root.left))
        if not root:
            return True
        if -1 <= height(root.right) - height(root.left) <= 1:
            return self.isBalanced(root.right) and self.isBalanced(root.left)
        return False


    def countNodes(self, root: Optional[TreeNode]) -> int:  # 222
        # go to leftest and rightest leaf and if the depths are the same return 2**(d+1)-1
        # else return 2**(d) + self.countNodes(root.left)
        if not root:
            return 0
        l, r = 0, 0
        root_left, root_right = root.left, root.right
        while root_left:
            if root_right:
                r+=1
                root_right = root_right.right
            l+=1
            root_left = root_left.left

        return 2**(l+1)-1 if l==r else 1 + self.countNodes(root.left) + self.countNodes(root.right)


    def getDecimalValue(self, head: ListNode) -> int:  # 1290
        result = 0
        while head:
            result *= 2
            result += head.val
            head = head.next

        return result


    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:  # 83
        #if head: if head.next: if head.next.val == head.val: head.next = head.next.next
        if not head or not head.next:
            return head
        result = head
        while head and head.next:
            if head.next.val == head.val:
                head.next = head.next.next
            else:
                head = head.next

        return result


    def threeSum(self, nums: List[int]) -> List[List[int]]:  # 15
        def two_sum(nums: List[int], target: int) -> List[List[int]]:
            remain = {}
            __result = []
            for i in nums:
                if remain.__contains__(i):
                    __result += [[i, target-i]]
                else:
                    remain[target-i]=None
            return __result

        result = []
        checked = {}
        for ind, i in enumerate(nums):  # O(n)
            if checked.__contains__(i):
                continue
            if x := two_sum(nums[ind+1:], -i):  # o(n)
                result+=[[i, t[0], t[1]] for t in x]
        return result


    def sqrt(self, x: int) -> int:  # 69
        # two pointer approach
        left, right = 0, x
        ans: int = -1  # ***
        while left <= right:
            mid  = (left + right) // 2
            if mid * mid == x:
                return mid
            elif mid * mid < x:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans

# 138
# # 315 hard
# # 84 hard
