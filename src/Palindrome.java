// palindrome or not
import java.util.*;

public class Palindrome{
 public static void main(String[] args){
    Scanner sc = new Scanner(System.in);
    int n= sc.nextInt();
    int rev=0;
    int temp=n;
    while(temp!=0){
        rev= rev*10+temp%10;
        temp=temp/10;
    }
    if(rev == n){
        System.out.println("Palindrome");
    }
    else{
        System.out.println("Not a Palindrome");
    }

 }   
}