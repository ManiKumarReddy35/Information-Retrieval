import java.util.Scanner;
class Methods
{
	int min(int a ,int b, int c)
	{
		if(a<=b && a<=c)
			return a;
		else if(b<a && b<c)
			return b;
		else 
			return c;
	}
}

public class MinEditDist 
{

	public static void main(String[] args) 
	{
		Methods m1 = new Methods();
		Scanner s = new Scanner(System.in);
		while(true) {
		System.out.println("\nSELECT ONE OF THE FOLLOWING CHOICES..\n1. DO YOU WANT TO ENTER TWO WORDS\n2. DO YOU WANT TO ENTER TWO SENTENCES\n3. EXIT");
		int ch = s.nextInt();
		
		if(ch==1)
		{
			
			System.out.print("ENTER THE WORD-1:");
			String a="";
			a+=s.next();
			System.out.print("ENTER THE WORD-2:");
			String b="";
			b+=s.next();
			int sizea= a.length();//4
			int sizeb = b.length();//5
			int[][] mat = new int[sizeb+1][sizea+1];
			int tempsizeb = sizeb;
			char[] a1 = a.toCharArray();
			char[] b1 = b.toCharArray();
			for(int i=0;i<sizeb;i++)
			{
				mat[i][0]=tempsizeb;
				tempsizeb--;
			}
			for(int i=0;i<sizea+1;i++)
				mat[sizeb][i] = i;
			// DONE WITH MATRIX INITIALIZATION...
		
			for(int i=sizeb-1;i>=0;i--)
			{
				for(int j=1;j<=sizea;j++)
				{
					if(b1[sizeb-1-i]!=a1[j-1])
						mat[i][j]=m1.min(mat[i+1][j]+1,mat[i][j-1]+1,mat[i+1][j-1]+2);
					if(b1[sizeb-1-i] == a1[j-1])
						mat[i][j]=m1.min(mat[i+1][j]+1,mat[i][j-1]+1,mat[i+1][j-1]);
				}
			}
		
			// DISPLAYING THE MINIMUM EDIT DISTANCE TABLE AFTER CALCULATING THE MINIMUM DISTANCES...
		
			System.out.print("\nMINIMUM EDIT DISTANCE MATRIX::\n");
		
		for(int i=0;i<sizeb+1;i++)
			 
		{
			System.out.println(" ");
			for(int j=0;j<sizea+1;j++)
				System.out.print(mat[i][j]+"   ");
		}
		System.out.print("\n\nMINIMUM EDIT DISTANCE::"+mat[0][sizea]);
		System.out.println(" ");
	   }
		
		else if(ch==2)
		{
			Scanner sc = new Scanner(System.in);
			
			
			System.out.print("ENTER THE SENTENCE-1::");
			String sent1 = sc.nextLine();
			String[] sent1_words = sent1.split(" ");
			int nwsent1 = sent1_words.length;
			//System.out.println("No. of words in sentence-1::"+nwsent1);
			
		    //NUMBER OF WORDS IN SENTENCE-1 IS IN "nwsent1"
		    
			System.out.print("ENTER THE SENTENCE-2::");
			String sent2 = sc.nextLine();
			String[] sent2_words= sent2.split(" ");
		    int nwsent2 = sent2_words.length;
		    //System.out.println("No. of words in sentence-2::"+nwsent2);
		    
		    //NUMBER OF WORDS IN SENTENCE-2 IS IN "nwsent2"
		    
		    int[][] mat =  new int[nwsent2+1][nwsent1+1];
		    int tempsizeb = nwsent2;
		    for(int i=0;i<nwsent2;i++)
			{
				mat[i][0]=tempsizeb;
				tempsizeb--;
			}
			for(int i=0;i<nwsent1+1;i++)
				mat[nwsent2][i] = i;
			
			// DONE WITH THE INITIALIZATION OF THE MATRIX
			
			for(int i=nwsent2-1;i>=0;i--)
			{
				for(int j=1;j<=nwsent1;j++)
				{
					if(sent2_words[nwsent2-1-i].equals(sent1_words[j-1]))
						mat[i][j]=m1.min(mat[i+1][j]+1,mat[i][j-1]+1,mat[i+1][j-1]);
					else
						mat[i][j]=m1.min(mat[i+1][j]+1,mat[i][j-1]+1,mat[i+1][j-1]+2);
				}
			}
			// DISPLAYING THE MINIMUM EDIT DISTANCE TABLE AFTER CALCULATING THE MINIMUM DISTANCES...
			
				System.out.print("\nMINIMUM EDIT DISTANCE MATRIX::\n");
				
			for(int i=0;i<nwsent2+1;i++)
			{
				System.out.println("");
				for(int j=0;j<nwsent1+1;j++)
				{
					System.out.print(mat[i][j]+" ");
				}
			}
			
			// DISPLAYING THE MINIMUM EDIT DISTANCE...
			
			System.out.print("\n\nTHE MINIMUM EDIT DISTANCE::"+mat[0][nwsent1]);
			System.out.println(" ");
		}
		
		else if(ch==3)
		{
			System.exit(0);
		}
		else
		{
			System.out.println("INVLAID CHOICE..");
		}
    }
  }
}
