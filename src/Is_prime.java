
import java.util.*;
// This program checks if a number is prime or not 
public class Is_prime{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter a number: ");
        int n;
        n=sc.nextInt();
        boolean prime= true;
        if(n<=1){
            prime=false;
            
        }
        for(int i=2; i<=Math.sqrt(n); i++){
            if(n%i==0){
                prime=false;
                break;
            }
        }
        if(prime){
            System.out.println(n + " is a prime number.");
        } else {
            System.out.println(n + " is not a prime number.");
        }
    }
}