import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class test {
    public void testMultiParams(String... params){
        for (String string : params) {
            System.out.println(string);
        }
    }


    public void testGener(List<?> list){}

    public String<Void> testGener1(){
        List<String> list  = new LinkedList<>();
        list.stream().map(System.out::println);

        int [] ints = new int[10];
        Arrays.stream(ints).map(e->{}).toArray();
    }
    
    public static void main(String[] args) {
        new test().testMultiParams("a","b","c");
    }
}
