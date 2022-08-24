[TOC]



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

