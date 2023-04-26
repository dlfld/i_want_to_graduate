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

    }
}
