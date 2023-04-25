import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class test {
    public void testMultiParams(String... params) {
        for (String string : params) {
            System.out.println(string);
        }
    }

    public void testGener(List<?> list) {
    }

    public void testGener1() {
        List<String> list = new LinkedList<>();
        // list.stream().map(System.out::println);

        int[] ints = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        int[] res = Arrays.stream(ints).toArray(int[]::new);
        // Arrays.<Object>asList("a", "b");
        // Arrays.stream(res).forEach(System.out::println);
    }

    public static void main(String[] args) {
        // new test().testMultiParams("a", "b", "c");
        new test().testGener1();
    }
}
