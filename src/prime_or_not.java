import java.util.*;

public class prime_or_not{
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        int num = sc.nextInt();

        boolean flag = num == 0 || num == 1;

        // 0 and 1 are not prime numbers
        // condition for nonprime number
        for (int i = 2; i <= num / 2; ++i)
            if (num % i == 0) {
                flag = true;
                break;
            }
        if (!flag)
            System.out.println(num + " is a prime number.");
        else
            System.out.println(num + " is not a prime number.");


    }
}