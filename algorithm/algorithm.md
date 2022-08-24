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

#### 6.盛水最多的容器
```
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。
![Image](https://github.com/yushen0/Gin/blob/main/algorithm/images/Container_with_the_most_water.jpg)
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

#### 7.三数之和

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

#### 8.电话号码的字母组合

```
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
![Image](https://github.com/yushen0/Gin/blob/main/algorithm/images/phoneNumber.jpg)

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


#### 9.删除链表的倒数第 N 个结点

```
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
![Image](https://github.com/yushen0/Gin/blob/main/algorithm/images/delete_N_ListNode.jpg)
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


#### 10.有效的括号

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
