[TOC]
- [1.两数之和](#1两数之和)
- [2.两数相加](#2两数相加)
- [3.无重复的最长字串（长度）](#3无重复的最长字串长度)
- [4.寻找两个正序数组的中位数](#4寻找两个正序数组的中位数)
- [5.最长回文子串](#5最长回文子串)
- [6.盛水最多的容器](#6盛水最多的容器)
- [7.三数之和](#7三数之和)
- [8.电话号码的字母组合](#8电话号码的字母组合)
- [9.删除链表的倒数第 N 个结点](#9删除链表的倒数第-n-个结点)
- [10.有效的括号](#10有效的括号)
- [11.合并两个有序链表](#11合并两个有序链表)
- [12.括号生成](#12括号生成)
- [13.合并K个升序链表](#13合并k个升序链表)
- [14.下一个排列](#14下一个排列)
- [15.最长有效括号](#15最长有效括号)
- [16.搜索旋转排序数组](#16搜索旋转排序数组)


##### 1.两数之和

```
两数之和：哈希表的使用，时间复杂度O(N)，N是数组的元素，因为要进行一次遍历
```

```java
public static int[] test() {
        int[] nums = {2, 7, 11, 15};
        int target = 26;

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0, len = nums.length; i < len; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[] {i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return new int[0];
    }
```



##### 2.两数相加

```
两数相加：链表的使用，需要注意进位，以及最后一个节点的进位
	时间复杂度：O(max(m,n))，其中 mm 和 nn 分别为两个链表的长度。我们要遍历两个链表的全部位置，而处理每个位置只需								要 O(1) 的时间。
	空间复杂度：O(1)。注意返回值不计入空间复杂度。
```

```java
public static ListNode test(ListNode listNode1, ListNode listNode2) {

        ListNode head = null, tail = null;
        int carry = 0;
        while (listNode1 != null || listNode2 != null) {
            int l1Val = listNode1 == null ? 0 : listNode1.val;
            int l2Val = listNode2 == null ? 0 : listNode2.val;

            int sum = l1Val + l2Val + carry;

            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }

            carry = sum / 10;

            if (listNode1 != null) {
                listNode1 = listNode1.next;
            }

            if (listNode2 != null) {
                listNode2 = listNode2.next;
            }

        }

        if (carry > 0) {
            tail.next = new ListNode(carry);
        }
        return head;
    }
```



##### 3.无重复的最长字串（长度）

```
哈希表的使用
	时间复杂度：O(N)
	一次遍历，将字符对应的坐标index存入哈希表，如果出现重复，就去比较重复字符位置的索引和start取最大值
```

```java
public static Integer test(String str) {
        HashMap<Character, Integer> map = new HashMap<>();
        int max = 0, start = 0;

        for (int end = 0; end < str.length(); end++) {
            char ch = str.charAt(end);
            if (map.containsKey(ch)) {
                start = Math.max(start, map.get(ch) + 1);
            }
            max = Math.max(max, end - start + 1);
            map.put(ch, end);
        }
        return max;
    }
```

##### 4.寻找两个正序数组的中位数

```
1.可以使用归并排序合并数组，找到中位数，时间复杂度O(m+n)
2.二分法的使用,时间复杂度O(log(m+n))

如果时间复杂度要求到log级别，通常是需要二分查找法

```
```java
/**
归并排序求解
**/
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        // 归并排序
        // 两个数组的起点
        int i = 0, j = 0;
        // 两个数组的终点
        int length1 = nums1.length, length2 = nums2.length;

        if(length1 == 0) {
            if (length2 % 2 == 1) {
                return nums2[length2 / 2];
            } else {
                return (nums2[(length2 /2) -1] + nums2[length2 /2]) / 2.0;
            }
        }

        if(length2 == 0) {
            if (length1 % 2 == 1) {
                return nums1[length1 / 2];
            } else {
                return (nums1[(length1 /2) -1] + nums1[length1 /2]) / 2.0;
            }
        }
        // 临时数组
        int[] tempArr = new int[length1+length2];
        // 临时数组索引
        int index = 0;
        while(i < length1 && j < length2){
            if (nums1[i] <= nums2[j]) {
                tempArr[index++] = nums1[i++];
            } else {
                tempArr[index++] = nums2[j++];
            }
        }

        if(i < length1){
            for (; i < length1; i++){
                tempArr[index++] = nums1[i];
            }
        }

        if(j < length2){
            for (; j < length2; j++){
                tempArr[index++] = nums2[j];
            }
        }

        
        int mid = (length1 + length2) / 2;
        if ((length1 + length2) % 2 == 1) {
            return tempArr[mid];
        } else {
            return (tempArr[mid-1] + tempArr[mid]) / 2.0;
        }
    }
}
```
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int totalLength = length1 + length2;
        if (totalLength % 2 == 1) {
            int midIndex = totalLength / 2;
            double median = getKthElement(nums1, nums2, midIndex + 1);
            return median;
        } else {
            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
            return median;
        }
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;
        int kthElement = 0;

        while (true) {
            // 边界情况
            if (index1 == length1) {
                return nums2[index2 + k - 1];
            }
            if (index2 == length2) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }
            
            // 正常情况
            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }
}
```

##### 5.最长回文子串

```
动态规划
时间复杂度：O(n^2))，其中 n 是字符串的长度。动态规划的状态总数为 O(n^2)，对于每个状态，我们需要转移的时间为 O(1)。
空间复杂度：O(n^2)，即存储动态规划状态需要的空间

涉及到动态规划，需要找到动态规划的状态转移方程，以及考虑动态规划的边界条件
	最优子结构、状态转移方程、边界、重叠子问题
注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，因此一定要注意动态规划的循环顺序。

动态规划的核心思想就是拆分子问题，记住过往，减少重复计算。 并且动态规划一般都是自底向上的
动态规划的思路：
	穷举分析
	确定边界
	找出规律，确定最优子结构
	写出状态转移方程
	
如果一个问题，可以把所有可能的答案穷举出来，并且穷举出来后，发现存在重叠子问题，就可以考虑使用动态规划。比如一些求最值的场景，如最长递增子序列、最小编辑距离、背包问题、凑零钱问题

一道动态规划问题，其实就是一个递推问题。假设当前决策结果是f(n),则最优子结构就是要让 f(n-k) 最优,最优子结构性质就是能让转移到n的状态是最优的,并且与后面的决策没有关系,即让后面的决策安心地使用前面的局部最优解的一种性质


```

```java
public static String test(String str) {

        // 确定边界
        if (str.length() < 2) {
            return str;
        }

        int begin = 0;
        int maxLen = 1;
        int length = str.length();
        // 备忘录
        boolean[][] dp = new boolean[length][length];
        for (int i = 0; i < length; i++) {
            // 最优子结构
            dp[i][i] = true;
        }

        char[] charArr = str.toCharArray();
        for (int L = 2; L < length; L++) {
            for (int i = 0; i < length; i++) {
                int j = L + i - 1;

                if (j >= length) {
                    break;
                }

                if (charArr[i] != charArr[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        // 状态转移方程
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }

        }
        return str.substring(begin, begin + maxLen);
    }
```

##### 6.盛水最多的容器
```
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。

双指针：
    理解涉及两边边界的问题，双指针解决
    需要计算体积，指针移动时，移动比较小的指针
    时间复杂度O(N),一次数组遍历
```

```java
class Solution {
    public int maxArea(int[] height) {
        int result = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            int maxArea = Math.min(height[left], height[right]) * (right - left);
            result = Math.max(result, maxArea);
            if (height[left] < height[right]){
                left++;
            } else {
                right--;
            }
        }
        return result;
    }
}
```

##### 7.三数之和

```
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

解题思路：
    排序+双指针，时间复杂度：O(N^2),因为双重循环

    排序是需要保证，三数相加=0，那么一定是 a < b < c,同时也是为了防止重复问题出现
    
    三数之和，其实也是两数之和的问题，只不过是target = 0 - nums[i], 外面一层循环，里面是判断两个数之和是否等于 target
    因为在循环中需要枚举两个元素进行判断，随着第一个元素递增，那么第二个元素一定是递减的，那么这种问题就可以用双指针解决
```

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        // 排序
        Arrays.sort(nums);

        for (int i=0, length = nums.length; i < length; i++) {
            // 如果第一个元素就大于0， 那么就不需要继续了
            if(nums[i] > 0) break;
            // 重复元素判断
            if(i>0 && nums[i] == nums[i-1]) continue;
            int target = -nums[i];

            int left = i + 1;
            int right = nums.length - 1;

            // 双指针
            while (left < right) {
                if (nums[left] + nums[right] == target) {
                    result.add(new ArrayList<>(Arrays.asList(nums[i], nums[left], nums[right])));
                    left++;
                    right--;
                    // 避免重复元素出现
                    while(left < right && nums[left] == nums[left-1]) left++;
                    while(left < right && nums[right] == nums[right+1]) right--;
                } else if (nums[left] + nums[right] < target){
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }
}
```

##### 8.电话号码的字母组合

```
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。


解题思路：回溯
    回溯算法用于寻找所有的可行解，如果发现一个解不可行，则会舍弃不可行的解。在这道题中，由于每个数字对应的每个字母都可能进入字母组合，因此不存在不可行的解，直接穷举所有的解即可。

    时间复杂度：O(3^m * 4^n)，其中 mm 是输入中对应 33 个字母的数字个数（包括数字 22、33、44、55、66、88），
    nn 是输入中对应 44 个字母的数字个数（包括数字 77、99），m+n 是输入数字的总个数。
    当输入包含 mm 个对应 33 个字母的数字和 nn 个对应 44 个字母的数字时，不同的字母组合一共有 3^m * 4^n种，需要遍历每一种字母组合。
```

```java
class Solution {

    // 数字到号码的映射
    private Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};

    // 路径
    private StringBuilder sb = new StringBuilder();
    // 结果集
    private List<String> res = new ArrayList<>();

    public List<String> letterCombinations(String digits) {
        if(digits == null || digits.length() == 0) return res;
        backtrack(digits,0);
        return res;
    }

    // 回溯函数
    private void backtrack(String digits,int index) {
        // 回溯的终止条件
        if(sb.length() == digits.length()) {
            res.add(sb.toString());
            return;
        }
        String value = phoneMap.get(digits.charAt(index));
        for(char ch:value.toCharArray()) {
            sb.append(ch);
            backtrack(digits,index+1);
            // 回溯完成需要把上次回溯的字母去掉
            sb.deleteCharAt(sb.length()-1);
        }
    }
}
```


##### 9.删除链表的倒数第 N 个结点

```
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

解题思路：
        1.一次遍历，找到链表的长度，我们首先从头节点开始对链表进行一次遍历，得到链表的长度 LL。随后我们再从头节点开始对链表进行一次遍历，当遍历到第 L-n+1L−n+1 个节点时，它就是我们需要删除的节点
        2.栈：我们也可以在遍历链表的同时将所有节点依次入栈。根据栈「先进后出」的原则，我们弹出栈的第 nn 个节点就是需要删除的节点，并且目前栈顶的节点就是待删除节点的前驱节点。
        3.双指针（也是快慢指针）由于我们需要找到倒数第 n 个节点，因此我们可以使用两个指针 {first} 和 {second} 同时对链表进行遍历，并且 {first} 比 {second} 超前 n 个节点。当 {first} 遍历到链表的末尾时，{second} 就恰好处于倒数第 nn 个节点。
        具体地，初始时 {first} 和 {second} 均指向头节点。我们首先使用 {first} 对链表进行遍历，遍历的次数为 n。此时，{first} 和 {second} 之间间隔了 n-1 个节点，即 {first} 比 {second} 超前了 n 个节点。
        在这之后，我们同时使用 {first} 和 {second} 对链表进行遍历。当 {first} 遍历到链表的末尾（即 {first} 为空指针）时，{second} 恰好指向倒数第 n 个节点。
        根据方法一和方法二，如果我们能够得到的是倒数第 n 个节点的前驱节点而不是倒数第 n 个节点的话，删除操作会更加方便。因此我们可以考虑在初始时将 {second} 指向哑节点，其余的操作步骤不变。这样一来，当 {first} 遍历到链表的末尾时，{second} 的下一个节点就是我们需要删除的节点。

        这里说一下哑节点（在head前创建一个新的节点，指向head）：为什么要在head前创建一个新的节点，这样做可以避免讨论头结点被删除的情况，不管原来的head有没有被删除，直接返回dummy.next即可

        三种解法的时间复杂度都是O(N),只需要一次遍历
```

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; ++i) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        ListNode ans = dummy.next;
        return ans;
    }
}
```


##### 10.有效的括号

```
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
有效字符串需满足：
    左括号必须用相同类型的右括号闭合。
    左括号必须以正确的顺序闭合。

解题思路：栈
        利用栈的先进后出的原理，入栈的一定是左边括号，右边括号跟栈顶元素是一对，也就是最接近的左边括号

        时间复杂度：o(N)，一次遍历
```

```java
class Solution {
    public boolean isValid(String s) {
        int n = s.length();
        // 有效的括号，字符串长度一定是偶数
        if(n % 2 == 1){
            return  false;
        }
        Map<Character, Character> map = new HashMap<Character, Character>() {{
            // 将 })] 作为key
            put('}', '{');
            put(']', '[');
            put(')', '(');
        }};
        // 新建一个栈
        LinkedList<Character> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            // 如果c是 })], 则判断， 否则说明是({[ , 直接入栈
            if(map.containsKey(c)){
                // stack.peek() 获取栈顶元素
                if(stack.isEmpty() || stack.peek() != map.get(c)){
                    return false;
                }
                // 将栈顶移除(先进后出，栈顶是最接近 c 的左括号)
                stack.pop();
            }else{
                // 说明c是({[ , 直接入栈
            stack.push(c);
        }
        }
        return stack.isEmpty();
    }
}
```

##### 11.合并两个有序链表

```
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

解题思路：
        1.循环迭代，比较两个链表节点的大小
        2.递归，代码简洁，思路就是以其中一个节点为头节点，用其next节点与另一个链表进行合并，边界条件就是链表到尾部，listNode == null(递归的要点就是一定要找到边界条件，否则会栈溢出)
        时间复杂度O(m+n)，m、n代表两个链表的长度
```

```
迭代循环
```
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {

        // 确定头节点指定一个哑节点
        ListNode head = new ListNode();
        ListNode current = head;
        while (list1 != null && list2 != null){
            if (list1.val <= list2.val) {
                current.next = list1; 
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }

        // 合并完成后，把剩下的节点接上
        current.next = list1 == null ? list2 : list1;
        return head.next;
    }
}
```

```
递归
```
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {

       if (list1 == null) {
           return list2;
       } 
       if (list2 == null) {
           return list1;
       }

       if (list1.val <= list2.val) {
           list1.next = mergeTwoLists(list1.next, list2);
           return list1;
       } else {
           list2.next = mergeTwoLists(list1, list2.next);
           return list2;
       }
    }
}
```

##### 12.括号生成

```
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
提示：1 <= n <= 8

解题思路：回溯&&递归
        主要思想，边界终止条件就是左右括号可以放置的个数都等于0
        否则就判断左右括号可以放置的数量大小
        回溯算法要注意一个细节，当左括号可以使用的个数严格大于右括号可以使用的个数时，不可以往下进行了，这种情况一定不会有有效的括号对生成了
        就是说无论怎么放置，剩下的可以使用的左边括号一定要小于等于可以使用的右边括号
```

```
递归
```
```java
class Solution {

    List<String> result = new ArrayList<>();
    public List<String> generateParenthesis(int n) {

        if (n == 0) {
            return result;
        }

        generateParenthesis("", n, n);
        return result;
    }

    public void generateParenthesis(String str, int left, int right) {
        if (left == 0 && right == 0) {
            result.add(str);
            return;
        }
      
        if (left == right) {
            // 代表左右括号目前放置的一样，现在就是有效的括号了，所以接下来只能放左括号
            generateParenthesis(str + "(", left-1, right);
        } else {
            // 代表目前既可以放左括号，又可以放右括号
            if (left > 0) {
                // 左括号还可以继续放
                generateParenthesis(str + "(", left-1, right);
            }
            // 只能放右括号
            generateParenthesis(str + ")", left, right-1);
        }
    }
}
```

```
回溯
```
```java
class Solution {

    List<String> result = new ArrayList<>();
    public List<String> generateParenthesis(int n) {

        if (n == 0) {
            return result;
        }

        generateParenthesis(new StringBuilder(), n, n);
        return result;
    }

    public void generateParenthesis(StringBuilder str, int left, int right) {
        if (left == 0 && right == 0) {
            result.add(str.toString());
            return;
        }

        // 左括号可以使用的个数严格大于右括号可以使用的个数
        if (left > right) {
            return;
        }

        if (left > 0) {
            generateParenthesis(str.append("("), left-1, right);
            str.deleteCharAt(str.length()-1);
        }

        if (right > 0) {
            generateParenthesis(str.append(")"), left, right-1);
            str.deleteCharAt(str.length()-1);
        }
    }
}

```

##### 13.合并K个升序链表

```
给你一个链表数组，每个链表都已经按升序排列。
请你将所有链表合并到一个升序链表中，返回合并后的链表。

解题思路：
        1.两两合并，将多个链表合并修改为两个链表的合并，时间复杂度较高
        2.两两合并，分治合并，可以使用二分法解决，时间复杂度：O(n * logk)
        2.优先级队列：PriorityQueue，即优先级队列。优先级队列可以保证每次取出来的元素都是队列中的最小或     最大的元素（Java优先级队列默认每次取出来的为最小元素）。

        底层原理：实际上优先级队列采用的是堆的形式来进行存储的，通过调整小根堆或大根堆来保证每次取出的元素为队列中最小或最大。
        小根堆（任意一个非叶子节点的权值，都不大于其左右子节点的权值）
        大根堆（任意一个非叶子节点的权值，都大于其左右子节点的权值）

        这样借助优先级队列，只需要将每个链表入队就可以，底层排好序后，再取出来，时间复杂度：O(n * logk)

```
```
两两合并
```
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        // 将题目改成两两合并
        ListNode listNode = null;
        for (int i = 0; i < lists.length; i++) {
            listNode = mergeTwoLists(listNode, lists[i]);
        }
        return listNode;
    }

    public ListNode mergeTwoLists(ListNode listNode1, ListNode listNode2) {
        // 定义一个哑节点
        ListNode head = new ListNode();
        ListNode current = head;

        while (listNode1 != null && listNode2 != null) {
            if (listNode1.val <= listNode2.val) {
                current.next = listNode1;
                listNode1 = listNode1.next;
            } else {
                current.next = listNode2;
                listNode2 = listNode2.next;
            }

            current = current.next;
        }

        current.next = listNode1 == null ? listNode2 : listNode1;
        return head.next;
    } 
}
```

```
分治合并
```
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        // 将题目改成两两合并
        if (lists == null || lists.length == 0){
            return null;
        }
        return merge(lists, 0, lists.length-1);
    }

    public ListNode merge(ListNode[] lists, int left, int right){
        // 递归终止条件，当左右坐标重合时，直接返回
        if (left == right) {
            return lists[left];
        }
        // 中间索引
        int mid = (right + left)/2;
        // 左边合并
        ListNode listNode1 = merge(lists, left, mid);
        // 右边合并
        ListNode listNode2 = merge(lists, mid+1, right);
        return mergeTwoLists(listNode1, listNode2);
    }

    public ListNode mergeTwoLists(ListNode listNode1, ListNode listNode2) {
        // 定义一个哑节点
        ListNode head = new ListNode();
        ListNode current = head;

        while (listNode1 != null && listNode2 != null) {
            if (listNode1.val <= listNode2.val) {
                current.next = listNode1;
                listNode1 = listNode1.next;
            } else {
                current.next = listNode2;
                listNode2 = listNode2.next;
            }

            current = current.next;
        }

        current.next = listNode1 == null ? listNode2 : listNode1;
        return head.next;
    } 
}

```
```
优先级队列
```
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
     PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for(int i = 0; i < lists.length; i++){
            if (lists[i] != null){
                priorityQueue.offer(lists[i]);
            }
        }
        ListNode head = new ListNode();
        ListNode tail = head;

        while(!priorityQueue.isEmpty()){
            ListNode current = priorityQueue.poll();
            tail.next = current;
            tail = tail.next;
            if (current.next != null) {
                priorityQueue.offer(current.next);
            }
        }
        return head.next;
    }
}

```

##### 14.下一个排列

```
整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。

例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
给你一个整数数组 nums ，找出 nums 的下一个排列。

必须 原地 修改，只允许使用额外常数空间。


题目解析：
    一个整数数组可以拼成一个数，例如[1,2,3]数组，可以拼成一个三位数 123
    举例 123 寻找下一个由这个整数数组拼成的，只比 123 大一点的数
    比如 123 的下一个 是 132，不是213
解题思路：
    先找出最大的索引 k 满足 nums[k] < nums[k+1]，如果不存在，就翻转整个数组；
    再找出另一个最大索引 l 满足 nums[l] > nums[k]；
    交换 nums[l] 和 nums[k]；
    最后翻转 nums[k+1:]。
    先找到满足nums[left] < nums[right]的索引 left，如果不存在就直接翻转数组，因为 321 的下一个题目要求是123
    再从右边【right,end】找到满足nums[left] < nums[index]的索引 index
    交换nums[left] 和 nums[index]
    交换完成后，[index+1, end] 范围内的数改为升序，就是直接反转过来就可以
    
```

```java
class Solution {
    public void nextPermutation(int[] nums) {
        int length = nums.length;
        //从后向前找第一次出现邻近升序的对儿，nums[leftIndex] < nums[rightIndex]
        int leftIndex = length - 2, rightIndex = length - 1;
        while(leftIndex >= 0) {
            if(nums[leftIndex] < nums[rightIndex]){
               break;
            }
            leftIndex--; 
            rightIndex--;
        }
       
        //没有下一个大的数，就返回最小的那个(升序的)，例如321的下一个就是最小的123,直接反转数组
        if(leftIndex < 0) {
            reverse(nums, 0, length-1);
            return;
        }
       
        //从[rightIndex, end]从后向前找第一个令nums[leftIndex] < nums[index]的 index索引对应的值
        int index;
        for(index = length-1; index >= rightIndex; index--){
            if(nums[leftIndex] < nums[index]) break;
        }

        //直接交换leftIndex, index
        swap(nums, leftIndex, index);
        //nums[rightIndex,end]是降序 需要改成为升序
        reverse(nums, rightIndex, length-1);
    }

    /**
     * 数组反转  
     */
    public void reverse(int[] nums, int leftIndex, int rightIndex){
        //双指针升序
        while(leftIndex < rightIndex){
            swap(nums, leftIndex, rightIndex);
            leftIndex++; 
            rightIndex--;
        }
    }
    /**
     *  两数交换
     */
    public void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
    
}
```

##### 15.最长有效括号

```
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
```

```java
```

##### 16.搜索旋转排序数组

```
整数数组 nums 按升序排列，数组中的值 互不相同 。
在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，
使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

解题思路：
    时间复杂度为 O(log n) ，第一反应是二分法
    主要点在于需要找到那个旋转的地方，所以比简单二分多加一个条件，找一下哪边是严格升序的部分
    需要注意的是，边界值的判断~

```

```java
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) {
            return -1;
        }
        if(nums.length == 1) {
            return nums[0] == target? 0 : -1;
        }
        return search(nums, 0, nums.length-1, target);
    }

    public int search(int[] nums, int start, int end, int target) {

        if (start > end) {
            return -1;
        }
        // 找到中间的索引
        int mid = (start + end) / 2;
        if (nums[mid] == target) {
            return mid;
        }

        if (nums[start] <= nums[mid]) {
            // 说明左边的是严格升序的 
            if (nums[start] <= target && target < nums[mid]) {
                // 说明目标值在左边，左边严格升序
                return search(nums, start, mid-1, target);
            } else {
                return search(nums, mid+1, end, target);
            }
        } else {
            // 说明右边是严格升序的
            if (nums[mid] < target && target <= nums[end]) {
                // 说明目标值在右边
                return search(nums, mid+1, end, target);
            } else {
                // 说明目标值在左边
                return search(nums, start, mid-1, target);
            }
        }        
    }

}
```

##### 17.在排序数组中查找元素的第一个和最后一个位置
```
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target，返回 [-1, -1]。
你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

解题思路：
         递增序列，查询目标值的开始和结束索引
         时间复杂度 O(logn),第一反应就是二分查找，找到开始的索引和结束的索引
         将问题改成二分寻找目标值的左右边界的问题
         可以单独找左边界，再单独找右边界

```

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        // 递增序列，查询目标值的开始和结束索引
        // 时间复杂度 O(logn),第一反应就是二分查找，找到开始的索引和结束的索引
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        // 改成二分寻找匹配目标值的左右边界的问题
        int startIndex = searchIndex(nums, 0, nums.length-1, target, true);
        int endIndex = searchIndex(nums, 0, nums.length-1, target, false);
        return new int[]{startIndex, endIndex};
    }

    public int searchIndex(int[] nums, int start, int end, int target, boolean isStart) {
        int index = -1;
        while(start <= end) {
            int mid = (start + end) / 2;
            if(target < nums[mid])
                end = mid - 1;
            else if(target > nums[mid])
                start = mid + 1;
            else {
                index = mid;
                //处理target == nums[mid]
                if(isStart){
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            }
        }
        return index;
    }
}
```

###### 17.组合总和
```
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
对于给定的输入，保证和为 target 的不同组合数少于 150 个。

解题思路：
    试图尝试各种组合的解题思路，我认为是回溯，本题的回溯有个问题，就是元素可以重复选取
    第一想法就是需要每个元素重复尝试到大于target，然后再和其他元素相加，直到排除该元素
    不符合就往后移动
```
```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // 试图尝试各种组合的解题思路，我认为是回溯，本题的回溯有个问题，就是元素可以重复选取
        // 第一想法就是需要每个元素重复尝试到大于target，然后再和其他元素相加，直到排除该元素
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        List<Integer> combination = new ArrayList<Integer>();
        backtrack(candidates, target, result, combination, 0);
        return result;
    }

    public void backtrack(int[] candidates, int target, List<List<Integer>> result, List<Integer> combination, int index) {
        // 回溯终止条件，走到黑的index
        if (index == candidates.length) {
            return;
        }

        // 符合条件的组合，添加进列表
        if (target == 0) {
            result.add(new ArrayList<>(combination));
            return;
        }
        
        // 直接跳过
        backtrack(candidates, target, result, combination, index + 1);
        // 选择当前数
        if (target - candidates[index] >= 0) {
            combination.add(candidates[index]);
            backtrack(candidates, target - candidates[index], result, combination, index);
            // 回溯路径终止返回后(成功或者不成功)，需要去除不能成功的路径
            combination.remove(combination.size() - 1);
        }
    }
}
```
